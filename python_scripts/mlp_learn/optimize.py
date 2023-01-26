import torch, time
from sdf.robot_sdf import RobotSdfCollisionNet

tensor_args = {'device': 'cpu', 'dtype': torch.float32}

def benchmark(model, x, n=1000):
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            y_pred = model(x)
    return (time.time()-t0)

q_dof = 7
data = torch.load('datasets/%d_dof_data_test.pt' % q_dof).to(**tensor_args)
x = data[:, 0:q_dof + 3]
y = data[:, -q_dof:]

s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (q_dof, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=x.shape[1], out_channels=y.shape[1], layers=[s] * n_layers, skips=skips)
nn_model.load_weights('models/' + fname, tensor_args)

model = nn_model.model
model.to(**tensor_args)

model = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

# backend = "fbgemm"
# model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# model = torch.quantization.prepare(model, inplace=False)
# model = torch.quantization.convert(model, inplace=False)

model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)

input_1 = x[0:1, :]
input_2 = x[:100, :]
t_tmp = benchmark(model, input_1)
t_tmp = benchmark(model, input_2)

t1 = benchmark(model, input_1)
t2 = benchmark(model, input_2)

print("Time for 1 sample: %4.2fms" % (1000*t1))
print("Time for 1000 samples: %4.2fms" % (1000*t2))
