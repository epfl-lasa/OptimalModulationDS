import torch, time
from sdf.robot_sdf import RobotSdfCollisionNet

from functorch import vmap, jacrev, vjp
from functorch.compile import aot_function, ts_compile
from torch.autograd.functional import jacobian
from sdf_transformer import *

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

# model = sdf_transformer(input_dim=x.shape[1], output_dim=y.shape[1],
#                         num_layer=2, embed_dim=128, nhead=4, ff_dim=256)
# chk = torch.load('models/t_' + fname)
# model.load_state_dict(chk["model_state_dict"])
model.to(**tensor_args)

# model = torch.quantization.quantize_dynamic(
#     model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
# )

# backend = "fbgemm"
# model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
model = torch.quantization.prepare(model, inplace=False)
model = torch.quantization.convert(model, inplace=False)
# model = sdf_transformer(input_dim=10, output_dim=7,
#                         num_layer=2, embed_dim=128, nhead=2, ff_dim=128)

model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)
# nn_model.model = model

input_1 = x[0:1, :]
input_2 = x[:100, :]
# t_tmp = benchmark(model, input_1)
# t_tmp = benchmark(model, input_2)
#
# t1 = benchmark(model, input_1)
# t2 = benchmark(model, input_2)
#
# print("Time for 1 sample: %4.2fms" % (1000*t1))
# print("Time for 1000 samples: %4.2fms" % (1000*t2))

n_iters = 100
nn_model.allocate_gradients(input_2.shape[0], tensor_args)
for i in range(n_iters):
    d1, g1, i1 = nn_model.dist_grad_closest(input_2)
    a = model.forward(input_2)

t0 = time.time()
for i in range(n_iters):
    d1, g1, i1 = nn_model.dist_grad_closest(input_2)
    #a = nn_model.model.forward(input_2)
    #a = model.forward(input_2)
print('Time for 1000 samples gradient: %4.2fmus' % (1e6*(time.time()-t0)/n_iters))


model2 = nn_model.model
def functorch_vjp(points):
    # def _vjp_fcn(points):
    #     dists, vjp_fn = vjp(model.forward, points)
    #     return torch.argmin(dists, dim=1), vjp_fn
    # out, fc = _vjp_fcn(points)
    dists, vjp_fn = vjp(model2.forward, points)
    minIdx = torch.argmin(dists, dim=1)
    grad_v = torch.zeros(points.shape[0], points.shape[1]-3)
    grad_v[list(range(points.shape[0])), minIdx] = 1
    return dists, vjp_fn(grad_v), minIdx


aot_lambda = aot_function(functorch_vjp, fw_compiler=ts_compile, bw_compiler=ts_compile)
for i in range(n_iters):
    d2, g2, i2 = aot_lambda(input_2)

t0 = time.time()
for i in range(n_iters):
    d2, g2, i2 = aot_lambda(input_2)
print('Time for 1000 samples gradient (AOT): %4.2fmus' % (1e6*(time.time()-t0)/n_iters))
print('fin')