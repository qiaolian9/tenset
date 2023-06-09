from tvm import te, auto_scheduler
import tvm

# a = te.placeholder((5,5))

# dim_var = [tvm.tir.IterVar((0, 5), 'x', 0), tvm.tir.IterVar((0, 5), 'y', 0)]
# fn = lambda i, j: a[i, j] + 1

# body = fn(*[v.var for v in dim_var])

# print(body, type(body))
@auto_scheduler.register_workload
def matmul_expr(shape, dataType="float32", for_rtile=False, pad={}):
    M, N, K = shape
    if for_rtile:
        return [("A", [K, M] ), ("B", [K, N])], [("compute", [M, N])]
    A = te.placeholder((K, M) , dtype=dataType, name="A")
    B = te.placeholder((K, N), dtype=dataType, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda y, x: te.sum((A[k, y]) * B[k, x], axis=k), name='compute')
    return [A, B, C]



# print(a.op, type(a.op.body), a.op.axis)
target = tvm.target.Target("cuda -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32")
shape = [128, 128, 128]

out = matmul_expr(shape)

print(out[2].op.body, type(out[2].op.body))
for i in out[2].op.body:
    print(type(i), type(i.source), i.source, i.axis[0].var.name)
    for j in i.source:
        print(j, '*' * 10)
        # print(j.indices[0].name, j.indices[0])
        print(j.a.indices, type(j.a), type(j))
        for k in j.a.indices:
            print(k, type(k))

task = tvm.auto_scheduler.SearchTask(func=matmul_expr, args=(shape, "float32"), target=target)
print("Computational DAG:")
print(task.compute_dag)

policy = auto_scheduler.search_policy.SketchPolicy(task)
policy.generate_sketches(True)
print("*" * 100)
populations = policy.sample_initial_population()
print(len(populations))
print(populations[0])
