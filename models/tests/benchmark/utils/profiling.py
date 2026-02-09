import torch

def benchmark(f, *inputs, mode="both", warmup_iter=100, run_iter=100):
    
    assert mode in {"fwd", "bwd", "both"}, "Invalid mode. Choose from 'fwd', 'bwd', or 'both'."
    
    torch._dynamo.reset() 
    cache_filler_1 = torch.empty((1024 * 1024 * 256), dtype=torch.int8, device="cuda:0")
    cache_filler_2 = torch.empty((1024 * 1024 * 256), dtype=torch.int8, device="cuda:0")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(1_000):
        cache_filler_1.zero_()
        cache_filler_2.zero_()
    end.record()
    torch.cuda.synchronize()

    filler_time = start.elapsed_time(end) / 2_000
    
    def dummy_loss(_out):
        loss = 0
        if isinstance(_out, torch.Tensor):
            loss += _out.sum()
        else:
            for o in _out:
                loss += o.sum()
        return loss
    
    def reset_grad(*inputs):
        for i in inputs:
            if isinstance(i, torch.Tensor) and i.requires_grad:
                i.grad = None
            elif isinstance(i, tuple):
                reset_grad(*i)
    
    if mode == "bwd":
        out = f(*inputs)
    for _ in range(warmup_iter):
        torch.compiler.cudagraph_mark_step_begin()
        if mode == "fwd" or mode == "both":
            out = f(*inputs)
        if mode == "bwd" or mode == "both":
            loss = dummy_loss(out)
            loss.backward()
            reset_grad(*inputs)
        cache_filler_1.zero_()
    
    if mode == "bwd":
        out = f(*inputs)
    start.record()
    for _ in range(run_iter):
        torch.compiler.cudagraph_mark_step_begin()
        if mode == "fwd" or mode == "both":
            out = f(*inputs)
        if mode == "bwd" or mode == "both":
            loss = dummy_loss(out)
            loss.backward()
            reset_grad(*inputs)
        cache_filler_1.zero_()
    end.record()
    torch.cuda.synchronize()
    
    run_time = start.elapsed_time(end) / run_iter - filler_time
    
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0") / (1024 ** 2)  # in MB
    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    
    try:
        print(f.__name__)
    except AttributeError:
        print(f.__class__.__name__)

    print(f"\telapsed time: {run_time:.2f} ms / iter")
    print(f"\tpeak memory usage: {peak_memory:.2f} MB")
    print()
    return run_time, peak_memory
