"""Host-memory probe for Megatron-LM runs (A/B host RSS regression hunt).

In-tree variant of the drop-in ``sitecustomize`` probe (imported from
``megatron/training/__init__.py``; inert unless ``MCORE_HOSTMEM_PROBE=1``): the probe
patches a few well-known functions lazily through an import hook and records, per
rank,

* anchors: ``probe_start``, ``model_build_begin``, ``kernel_warmup_done`` (if the tree
  has it), ``optimizer_built``, ``stepN_begin``, ``stepN_first_forward``,
  ``stepN_fwdbwd_done``, ``stepN_optim_begin``, ``offloader_step_begin/end``,
  ``first_newton_schulz``, ``stepN_optim_end``, ``stepN_end`` (N <= MCORE_HOSTMEM_PROBE_STEPS,
  default 3) and ``exit``;
* at every anchor: ``/proc/self/status`` (VmRSS/VmHWM/VmLck/VmPin/VmSwap/RssAnon/RssFile/
  RssShmem), ``/proc/meminfo`` (node), the memory cgroup (``memory.current/peak/max``),
  ``torch.cuda.host_memory_stats()`` (pinned allocated/reserved current+peak),
  device allocated/reserved/max, and the Python-visible pinned allocations since the
  previous anchor attributed to their call stacks;
* a sampler thread (every MCORE_HOSTMEM_PROBE_INTERVAL seconds, default 5) that appends
  the same process/node numbers, so an OOM kill leaves a timeline behind.

Pinned allocations are attributed by wrapping ``torch.{empty,zeros,ones,full,empty_strided,
tensor,*_like}(pin_memory=True)`` and ``torch.Tensor.pin_memory`` (bytes requested, count,
and live bytes via weakref finalizers).  Allocations made from C++ (NCCL, TE, cudaHostAlloc)
are not attributed but show up in VmLck/VmPin and in the node ``Mlocked``/``Unevictable``.

Output: ``$MCORE_HOSTMEM_PROBE_DIR`` (os.path.expandvars at run time, so ``$SLURM_JOB_ID``
works) / ``rank<RANK>_<host>.jsonl``; every rank also prints one ``[HOSTMEM]`` line per
anchor to stdout (interleaves with ``NCCL_DEBUG=INFO`` lines under ``srun -l``).
Every hook is wrapped in try/except: a probe failure never changes training behaviour.
"""

from __future__ import annotations

import atexit
import builtins
import collections
import json
import os
import socket
import sys
import threading
import time
import traceback
import weakref

_ENABLED = os.environ.get("MCORE_HOSTMEM_PROBE") == "1"
_STEPS = int(os.environ.get("MCORE_HOSTMEM_PROBE_STEPS", "3") or 3)
_INTERVAL = float(os.environ.get("MCORE_HOSTMEM_PROBE_INTERVAL", "5") or 5)
_STACK_DEPTH = int(os.environ.get("MCORE_HOSTMEM_PROBE_STACK_DEPTH", "8") or 8)
_TOP_STACKS = int(os.environ.get("MCORE_HOSTMEM_PROBE_TOP_STACKS", "25") or 25)
_T0 = time.time()
_THIS_FILE = os.path.abspath(__file__)

_lock = threading.Lock()
_state = {
    "rank": int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")) or 0),
    "local_rank": int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")) or 0),
    "host": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
    "step": 0,
    "anchors": 0,
    "fh": None,
    "ns_seen": False,
    "first_forward_seen_step": -1,
    "pinned_cum_by_stack": collections.Counter(),
    "pinned_cnt_by_stack": collections.Counter(),
    "pinned_live_by_stack": collections.Counter(),
    "pinned_cum_total": 0,
    "pinned_live_total": 0,
    "pinned_cum_at_last_anchor": collections.Counter(),
    "pinned_cum_total_at_last_anchor": 0,
    "patched": set(),
    # Triton OutOfResources exceptions (autotune configs over the shared-memory limit)
    "oor_count": 0,
    "oor_first_t": None,
    "oor_last_t": None,
    # fine-grained activation offload: live pools / managers / handlers and per-interval counters
    "pools": weakref.WeakSet(),
    "managers": weakref.WeakSet(),
    "handlers": weakref.WeakSet(),
    "offload_calls": 0,
    "offload_view_bytes": 0,
    "offload_host_bytes": 0,
    "offload_storage_path_calls": 0,
    "reload_calls": 0,
    "reload_dev_delta_bytes": 0,
    "reload_out_bytes": 0,
    # gc activity seen by the gc.callbacks hook
    "gc_events": 0,
    "gc_collected": 0,
    "gc_last": None,
}
_CENSUS_AT = {s for s in os.environ.get(
    "MCORE_HOSTMEM_PROBE_CENSUS_AT",
    "optimizer_built,step1_fwdbwd_done,step1_end,step2_end,step3_end").split(",") if s}
_CENSUS_REFERRER_RANKS = {int(s) for s in os.environ.get("MCORE_HOSTMEM_PROBE_CENSUS_RANKS", "0,64,128,192").split(",") if s.strip().isdigit()}
_CENSUS_REFERRER_TENSORS = int(os.environ.get("MCORE_HOSTMEM_PROBE_CENSUS_REFERRERS", "12") or 12)
_GC_AT = {s for s in os.environ.get("MCORE_HOSTMEM_PROBE_GC_AT", "").split(",") if s}


# --------------------------------------------------------------------------- readers
def _kb_fields(path, wanted):
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in wanted:
                    parts = rest.split()
                    if parts:
                        try:
                            value = int(parts[0])
                        except ValueError:
                            continue
                        if len(parts) > 1 and parts[1] == "kB":
                            value *= 1024
                        out[key] = value
    except Exception:  # pylint: disable=broad-except
        pass
    return out


_STATUS_KEYS = {"VmRSS", "VmHWM", "VmLck", "VmPin", "VmSwap", "RssAnon", "RssFile", "RssShmem", "Threads"}
_MEMINFO_KEYS = {
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem", "Unevictable", "Mlocked",
    "AnonPages", "Mapped", "Slab", "HugePages_Total", "HugePages_Free", "Hugepagesize",
}


def _read_status():
    return _kb_fields("/proc/self/status", _STATUS_KEYS)


def _read_meminfo():
    return _kb_fields("/proc/meminfo", _MEMINFO_KEYS)


def _read_cgroup():
    out = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.high"):
        try:
            with open(f"/sys/fs/cgroup/{name}") as fh:
                raw = fh.read().strip()
            out[name] = raw if raw == "max" else int(raw)
        except Exception:  # pylint: disable=broad-except
            continue
    if not out:
        for name, key in (("memory.usage_in_bytes", "memory.current"), ("memory.max_usage_in_bytes", "memory.peak"),
                          ("memory.limit_in_bytes", "memory.max")):
            try:
                with open(f"/sys/fs/cgroup/memory/{name}") as fh:
                    out[key] = int(fh.read().strip())
            except Exception:  # pylint: disable=broad-except
                continue
    return out


def _read_proc_stat(pid):
    """Return (utime+stime seconds, ppid, comm) of a pid from /proc/<pid>/stat, or None."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            raw = fh.read()
        rpar = raw.rindex(")")
        comm = raw[raw.index("(") + 1:rpar]
        fields = raw[rpar + 2:].split()
        ppid = int(fields[1])
        cpu_s = (int(fields[11]) + int(fields[12])) / _CLK_TCK
        return cpu_s, ppid, comm
    except Exception:  # pylint: disable=broad-except
        return None


try:
    _CLK_TCK = os.sysconf("SC_CLK_TCK") or 100
except Exception:  # pylint: disable=broad-except
    _CLK_TCK = 100


def _read_children():
    """Descendant processes of this rank (Inductor/compile worker pools, dataloader workers, ...).

    Slurm's per-task RSS is the sum over the task's cgroup, so children count towards the
    accounting even though they are invisible in the main process' VmRSS.
    """
    me = os.getpid()
    parent_of = {}
    info = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            stat = _read_proc_stat(pid)
            if stat is None:
                continue
            cpu_s, ppid, comm = stat
            parent_of[pid] = ppid
            info[pid] = (cpu_s, comm)
    except Exception:  # pylint: disable=broad-except
        return {}
    children = {}
    for pid, ppid in parent_of.items():
        children.setdefault(ppid, []).append(pid)
    descendants = []
    stack = list(children.get(me, []))
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children.get(pid, []))
    rss_total = 0
    cpu_total = 0.0
    names = collections.Counter()
    biggest = []
    for pid in descendants:
        status = _kb_fields(f"/proc/{pid}/status", {"VmRSS"})
        rss = status.get("VmRSS", 0)
        rss_total += rss
        cpu_s, comm = info.get(pid, (0.0, "?"))
        cpu_total += cpu_s
        names[comm] += 1
        biggest.append((rss, comm, pid))
    biggest.sort(reverse=True)
    own = _read_proc_stat(me)
    return {
        "child_count": len(descendants),
        "child_rss": rss_total,
        "child_cpu_s": round(cpu_total, 1),
        "child_names": dict(names.most_common(6)),
        "child_top": [{"rss": r, "comm": c, "pid": p} for r, c, p in biggest[:5]],
        "self_cpu_s": round(own[0], 1) if own else None,
    }


def _gc_stats():
    try:
        import gc  # noqa: PLC0415

        return {"enabled": gc.isenabled(), "count": list(gc.get_count()),
                "collections": [s.get("collections") for s in gc.get_stats()],
                "events_seen": _state["gc_events"], "collected_seen": _state["gc_collected"]}
    except Exception:  # pylint: disable=broad-except
        return {}


def _jsonable(value, depth=0):
    """Best-effort conversion of pool-status / attribute values to JSON types."""
    if depth > 4:
        return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:64]]
    return str(value)


def _offload_state():
    """Fine-grained offload pools (status), managers (policy knobs) and interval counters."""
    out = {
        "offload_calls": _state["offload_calls"],
        "offload_view_bytes": _state["offload_view_bytes"],
        "offload_host_bytes": _state["offload_host_bytes"],
        "offload_storage_path_calls": _state["offload_storage_path_calls"],
        "reload_calls": _state["reload_calls"],
        "reload_dev_delta_bytes": _state["reload_dev_delta_bytes"],
        "reload_out_bytes": _state["reload_out_bytes"],
        "pools": [],
        "managers": [],
    }
    try:
        for pool in list(_state["pools"]):
            entry = {"device": str(getattr(pool, "device", "?")), "pin_memory": getattr(pool, "pin_memory", None)}
            try:
                entry["status"] = _jsonable(pool.get_pool_status())
            except Exception as exc:  # pylint: disable=broad-except
                entry["status_error"] = f"{type(exc).__name__}: {exc}"[:120]
            # per-shape pools: count / in use / bytes, independent of get_pool_status's format
            pools = getattr(pool, "_pools", None)
            if isinstance(pools, dict):
                per_shape = []
                total_bytes = 0
                total_blocks = 0
                for key, bucket in list(pools.items())[:64]:
                    blocks = bucket.get("all", []) if isinstance(bucket, dict) else []
                    free = bucket.get("free", []) if isinstance(bucket, dict) else []
                    nbytes = 0
                    for t in blocks[:1]:
                        try:
                            nbytes = int(t.untyped_storage().nbytes())
                        except Exception:  # pylint: disable=broad-except
                            nbytes = 0
                    total_bytes += nbytes * len(blocks)
                    total_blocks += len(blocks)
                    per_shape.append({"key": str(key)[:80], "blocks": len(blocks), "free": len(free), "block_bytes": nbytes})
                entry["per_shape"] = per_shape
                entry["total_blocks"] = total_blocks
                entry["total_bytes"] = total_bytes
            out["pools"].append(entry)
        for mgr in list(_state["managers"]):
            entry = {}
            for attr in ("_offload_margin", "max_inflight_offloads", "keep_on_gpu_bytes", "delta_offload_bytes_across_pp_ranks",
                         "num_offload_group", "_num_offload_group", "pp_rank", "_pp_rank"):
                if hasattr(mgr, attr):
                    entry[attr] = _jsonable(getattr(mgr, attr))
            for meth in ("offload_summary_bytes", "offload_summary_total_bytes"):
                fn = getattr(mgr, meth, None)
                if callable(fn):
                    try:
                        entry[meth] = _jsonable(fn())
                    except Exception:  # pylint: disable=broad-except
                        pass
            out["managers"].append(entry)
        handlers = list(_state["handlers"])
        out["handlers"] = len(handlers)
        knobs = collections.Counter()
        for h in handlers[:256]:
            for attr in ("max_inflight_offloads", "keep_on_gpu_bytes", "_offload_margin"):
                if hasattr(h, attr):
                    knobs[f"{attr}={_jsonable(getattr(h, attr))}"] += 1
        out["handler_knobs"] = dict(knobs.most_common(12))
    except Exception as exc:  # pylint: disable=broad-except
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def _pinned_census(with_referrers):
    """Count CPU pinned tensors reachable from the gc heap; optionally who refers to them."""
    import gc  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415
    except Exception:  # pylint: disable=broad-except
        return {}
    out = {"tensors": 0, "unique_storages": 0, "bytes": 0}
    seen = set()
    pinned = []
    try:
        for obj in gc.get_objects():
            try:
                if type(obj).__module__ != "torch" and not isinstance(obj, torch.Tensor):
                    continue
                if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu" or not obj.is_pinned():
                    continue
            except Exception:  # pylint: disable=broad-except
                continue
            out["tensors"] += 1
            try:
                storage = obj.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
            except Exception:  # pylint: disable=broad-except
                continue
            if key not in seen:
                seen.add(key)
                out["bytes"] += key[1]
                pinned.append(obj)
        out["unique_storages"] = len(seen)
        pinned.sort(key=lambda t: -_safe_nbytes(t))
        out["largest"] = [{"shape": list(t.shape), "dtype": str(t.dtype), "bytes": _safe_nbytes(t)} for t in pinned[:8]]
        if with_referrers:
            refs = []
            for t in pinned[:_CENSUS_REFERRER_TENSORS]:
                level1 = collections.Counter()
                level2 = collections.Counter()
                for r in gc.get_referrers(t)[:12]:
                    if r is pinned or r is refs:
                        continue
                    level1[_describe(r)] += 1
                    for rr in gc.get_referrers(r)[:8]:
                        if rr is pinned or rr is refs:
                            continue
                        level2[_describe(rr)] += 1
                refs.append({"bytes": _safe_nbytes(t), "shape": list(t.shape), "referrers": dict(level1.most_common(6)),
                             "referrers2": dict(level2.most_common(6))})
            out["referrers"] = refs
    except Exception as exc:  # pylint: disable=broad-except
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def _safe_nbytes(t):
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:  # pylint: disable=broad-except
        return 0


def _describe(obj):
    import types  # noqa: PLC0415

    try:
        if isinstance(obj, types.FrameType):
            return f"frame:{os.path.basename(obj.f_code.co_filename)}:{obj.f_lineno}:{obj.f_code.co_name}"
        if isinstance(obj, types.TracebackType):
            return f"traceback:{os.path.basename(obj.tb_frame.f_code.co_filename)}:{obj.tb_frame.f_code.co_name}"
        if isinstance(obj, dict):
            keys = list(obj.keys())[:3]
            return f"dict[{','.join(str(k)[:20] for k in keys)}]"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}[{len(obj)}]"
        if isinstance(obj, types.CellType):
            return "cell"
        return type(obj).__name__
    except Exception:  # pylint: disable=broad-except
        return "?"


def _torch_stats():
    out = {}
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_initialized():
            return out
        out["dev_alloc"] = int(torch.cuda.memory_allocated())
        out["dev_max_alloc"] = int(torch.cuda.max_memory_allocated())
        out["dev_reserved"] = int(torch.cuda.memory_reserved())
        hms = getattr(torch.cuda, "host_memory_stats", None) or getattr(torch.cuda.memory, "host_memory_stats", None)
        if hms is not None:
            stats = hms()
            for key, name in (
                ("allocated_bytes.all.current", "pin_alloc"),
                ("allocated_bytes.all.peak", "pin_alloc_peak"),
                ("reserved_bytes.all.current", "pin_reserved"),
                ("reserved_bytes.all.peak", "pin_reserved_peak"),
                ("num_alloc_retries", "pin_alloc_retries"),
            ):
                if key in stats:
                    out[name] = int(stats[key])
    except Exception:  # pylint: disable=broad-except
        pass
    return out


# --------------------------------------------------------------------------- output
def _out_dir():
    raw = os.environ.get("MCORE_HOSTMEM_PROBE_DIR") or os.path.join(os.getcwd(), "hostmem_probe")
    return os.path.expandvars(raw)


def _write(record):
    try:
        with _lock:
            fh = _state["fh"]
            if fh is None:
                directory = _out_dir()
                os.makedirs(directory, exist_ok=True)
                path = os.path.join(directory, f"rank{_state['rank']:05d}_{_state['host']}.jsonl")
                fh = open(path, "a", buffering=1)  # noqa: SIM115
                _state["fh"] = fh
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            fh.flush()
    except Exception:  # pylint: disable=broad-except
        pass


def _gib(value):
    try:
        return f"{value / 2**30:.2f}"
    except Exception:  # pylint: disable=broad-except
        return "?"


def _snapshot(kind, name, extra=None):
    status = _read_status()
    meminfo = _read_meminfo()
    cgroup = _read_cgroup()
    tstats = _torch_stats()
    children = _read_children()
    with _lock:
        cum_total = _state["pinned_cum_total"]
        live_total = _state["pinned_live_total"]
    record = {
        "type": kind,
        "name": name,
        "t": round(time.time() - _T0, 3),
        "wall": time.time(),
        "rank": _state["rank"],
        "local_rank": _state["local_rank"],
        "host": _state["host"],
        "step": _state["step"],
        "status": status,
        "meminfo": meminfo,
        "cgroup": cgroup,
        "torch": tstats,
        "children": children,
        "pinned_py_cum": cum_total,
        "pinned_py_live": live_total,
        "oor": {"count": _state["oor_count"], "first_t": _state["oor_first_t"], "last_t": _state["oor_last_t"]},
        "gc": _gc_stats(),
    }
    if kind == "anchor":
        record["offload"] = _offload_state()
    if extra:
        record.update(extra)
    return record


# --------------------------------------------------------------------------- pinned attribution
def _stack_sig():
    frames = traceback.extract_stack(limit=60)
    keep = []
    for fr in frames:
        fn = fr.filename or ""
        if fn == _THIS_FILE or "/torch/" in fn or fn.startswith("<"):
            continue
        short = fn
        for marker in ("/megatron/", "/transformer_engine/", "/emerging_optimizers/", "/fla/", "/mok/", "/examples/"):
            idx = fn.find(marker)
            if idx >= 0:
                short = fn[idx + 1:]
                break
        else:
            short = "/".join(fn.split("/")[-2:])
        keep.append(f"{short}:{fr.lineno}:{fr.name}")
    return " < ".join(reversed(keep[-_STACK_DEPTH:]))


def _on_free(sig, nbytes):
    with _lock:
        _state["pinned_live_by_stack"][sig] -= nbytes
        _state["pinned_live_total"] -= nbytes


def _record_pinned(tensor):
    try:
        nbytes = int(tensor.untyped_storage().nbytes())
    except Exception:  # pylint: disable=broad-except
        try:
            nbytes = int(tensor.numel() * tensor.element_size())
        except Exception:  # pylint: disable=broad-except
            return
    sig = _stack_sig()
    with _lock:
        _state["pinned_cum_by_stack"][sig] += nbytes
        _state["pinned_cnt_by_stack"][sig] += 1
        _state["pinned_live_by_stack"][sig] += nbytes
        _state["pinned_cum_total"] += nbytes
        _state["pinned_live_total"] += nbytes
    try:
        weakref.finalize(tensor, _on_free, sig, nbytes)
    except Exception:  # pylint: disable=broad-except
        pass


def _patch_pinned_factories():
    try:
        import torch  # noqa: PLC0415
    except Exception:  # pylint: disable=broad-except
        return
    for fname in ("empty", "zeros", "ones", "full", "empty_strided", "tensor", "empty_like", "zeros_like",
                  "ones_like", "full_like", "randn", "rand"):
        original = getattr(torch, fname, None)
        if original is None or getattr(original, "_hostmem_patched", False):
            continue

        def wrapper(*args, __original=original, **kwargs):
            out = __original(*args, **kwargs)
            if kwargs.get("pin_memory"):
                try:
                    _record_pinned(out)
                except Exception:  # pylint: disable=broad-except
                    pass
            return out

        wrapper._hostmem_patched = True
        wrapper.__name__ = getattr(original, "__name__", fname)
        wrapper.__doc__ = getattr(original, "__doc__", None)
        try:
            setattr(torch, fname, wrapper)
        except Exception:  # pylint: disable=broad-except
            continue
    original_pin = getattr(torch.Tensor, "pin_memory", None)
    if original_pin is not None and not getattr(original_pin, "_hostmem_patched", False):

        def pin_memory(self, *args, __original=original_pin, **kwargs):
            try:
                was_pinned = self.is_pinned()
            except Exception:  # pylint: disable=broad-except
                was_pinned = False
            out = __original(self, *args, **kwargs)
            if not was_pinned:
                try:
                    _record_pinned(out)
                except Exception:  # pylint: disable=broad-except
                    pass
            return out

        pin_memory._hostmem_patched = True
        try:
            torch.Tensor.pin_memory = pin_memory
        except Exception:  # pylint: disable=broad-except
            pass


# --------------------------------------------------------------------------- anchors
def anchor(name, extra=None):
    """Record an anchor: JSONL record + pinned stack deltas + one stdout line."""
    try:
        record = _snapshot("anchor", name, extra)
        with _lock:
            _state["anchors"] += 1
            cum = _state["pinned_cum_by_stack"]
            last = _state["pinned_cum_at_last_anchor"]
            delta = collections.Counter()
            for sig, value in cum.items():
                diff = value - last.get(sig, 0)
                if diff:
                    delta[sig] = diff
            delta_total = _state["pinned_cum_total"] - _state["pinned_cum_total_at_last_anchor"]
            _state["pinned_cum_at_last_anchor"] = collections.Counter(cum)
            _state["pinned_cum_total_at_last_anchor"] = _state["pinned_cum_total"]
            stacks = [
                {"stack": sig, "delta_bytes": value, "cum_bytes": cum[sig],
                 "live_bytes": _state["pinned_live_by_stack"][sig], "count": _state["pinned_cnt_by_stack"][sig]}
                for sig, value in delta.most_common(_TOP_STACKS)
            ]
        record["pinned_py_delta"] = delta_total
        _write(record)
        if stacks:
            _write({"type": "pinned_stacks", "name": name, "rank": _state["rank"], "step": _state["step"],
                    "t": record["t"], "stacks": stacks})
        if name in _CENSUS_AT:
            t0 = time.time()
            census = _pinned_census(with_referrers=_state["rank"] in _CENSUS_REFERRER_RANKS)
            census["seconds"] = round(time.time() - t0, 2)
            _write({"type": "census", "name": name, "rank": _state["rank"], "step": _state["step"], "t": record["t"],
                    "census": census})
        if name in _GC_AT:
            import gc  # noqa: PLC0415

            before = _snapshot("gc_probe_before", name)
            t0 = time.time()
            try:
                import torch  # noqa: PLC0415

                torch.cuda.synchronize()
            except Exception:  # pylint: disable=broad-except
                pass
            collected = gc.collect()
            after = _snapshot("gc_probe_after", name)
            after["gc_collected"] = collected
            after["gc_seconds"] = round(time.time() - t0, 2)
            _write(before)
            _write(after)
            print(f"[HOSTMEM] gc_probe anchor={name} rank={_state['rank']} collected={collected} "
                  f"rss {_gib(before['status'].get('VmRSS', 0))} -> {_gib(after['status'].get('VmRSS', 0))} "
                  f"pin_alloc {_gib(before['torch'].get('pin_alloc', 0))} -> {_gib(after['torch'].get('pin_alloc', 0))} "
                  f"pin_resv {_gib(before['torch'].get('pin_reserved', 0))} -> {_gib(after['torch'].get('pin_reserved', 0))} "
                  f"dev_alloc {_gib(before['torch'].get('dev_alloc', 0))} -> {_gib(after['torch'].get('dev_alloc', 0))}",
                  flush=True)
        st, mi, tt, cg, ch = record["status"], record["meminfo"], record["torch"], record["cgroup"], record["children"]
        pools = record.get("offload", {}).get("pools", [])
        pool_bytes = sum(p.get("total_bytes", 0) for p in pools)
        pool_blocks = sum(p.get("total_blocks", 0) for p in pools)
        print(
            f"[HOSTMEM] anchor={name} rank={_state['rank']} host={_state['host']} t={record['t']:.0f}s "
            f"rss={_gib(st.get('VmRSS', 0))} hwm={_gib(st.get('VmHWM', 0))} lck={_gib(st.get('VmLck', 0))} "
            f"pin={_gib(st.get('VmPin', 0))} anon={_gib(st.get('RssAnon', 0))} "
            f"pin_alloc={_gib(tt.get('pin_alloc', 0))} pin_resv={_gib(tt.get('pin_reserved', 0))} "
            f"pin_py_cum={_gib(record['pinned_py_cum'])} pin_py_live={_gib(record['pinned_py_live'])} "
            f"pin_py_delta={_gib(delta_total)} dev_alloc={_gib(tt.get('dev_alloc', 0))} "
            f"children={ch.get('child_count', 0)} child_rss={_gib(ch.get('child_rss', 0))} "
            f"child_cpu_s={ch.get('child_cpu_s', 0)} self_cpu_s={ch.get('self_cpu_s', 0)} "
            f"pool_blocks={pool_blocks} pool_bytes={_gib(pool_bytes)} offload_calls={_state['offload_calls']} "
            f"offload_host={_gib(_state['offload_host_bytes'])} offload_view={_gib(_state['offload_view_bytes'])} "
            f"storage_path={_state['offload_storage_path_calls']} reload_calls={_state['reload_calls']} "
            f"reload_dev_delta={_gib(_state['reload_dev_delta_bytes'])} oor={_state['oor_count']} "
            f"gc_events={_state['gc_events']} "
            f"node_avail={_gib(mi.get('MemAvailable', 0))} node_mlocked={_gib(mi.get('Mlocked', 0))} "
            f"node_unevict={_gib(mi.get('Unevictable', 0))} cg_cur={_gib(cg.get('memory.current', 0)) if isinstance(cg.get('memory.current'), int) else '?'}",
            flush=True,
        )
    except Exception:  # pylint: disable=broad-except
        pass


def _sampler():
    while True:
        time.sleep(_INTERVAL)
        try:
            _write(_snapshot("sample", "periodic"))
        except Exception:  # pylint: disable=broad-except
            pass


# --------------------------------------------------------------------------- lazy patches
def _wrap_function(module, attr, before=None, after=None, tag=None):
    original = getattr(module, attr, None)
    if original is None:
        return False
    if getattr(original, "_hostmem_patched", False):
        return True

    def wrapper(*args, __original=original, **kwargs):
        if before is not None:
            try:
                before(*args, **kwargs)
            except Exception:  # pylint: disable=broad-except
                pass
        result = __original(*args, **kwargs)
        if after is not None:
            try:
                after(result, *args, **kwargs)
            except Exception:  # pylint: disable=broad-except
                pass
        return result

    wrapper._hostmem_patched = True
    wrapper.__name__ = getattr(original, "__name__", attr)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    wrapper.__wrapped__ = original
    setattr(module, attr, wrapper)
    _state["patched"].add(tag or f"{getattr(module, '__name__', module)}.{attr}")
    return True


def _step_label(suffix):
    return f"step{_state['step']}_{suffix}"


def _in_window():
    return 1 <= _state["step"] <= _STEPS


def _patch_training(module):
    def before_setup(*args, **kwargs):
        anchor("model_build_begin")

    def after_setup(result, *args, **kwargs):
        anchor("optimizer_built")

    ok = _wrap_function(module, "setup_model_and_optimizer", before=before_setup, after=after_setup)

    def before_train_step(*args, **kwargs):
        _state["step"] += 1
        if _in_window():
            anchor(_step_label("begin"))

    def after_train_step(result, *args, **kwargs):
        if _in_window():
            anchor(_step_label("end"))

    return _wrap_function(module, "train_step", before=before_train_step, after=after_train_step) and ok


def _patch_kernel_warmup(module):
    def after_warmup(result, *args, **kwargs):
        anchor("kernel_warmup_done")

    return _wrap_function(module, "warmup_training_kernels", after=after_warmup)


def _patch_schedules(module):
    def before_forward_step(*args, **kwargs):
        if _in_window() and _state["first_forward_seen_step"] != _state["step"]:
            _state["first_forward_seen_step"] = _state["step"]
            anchor(_step_label("first_forward"))

    ok = _wrap_function(module, "forward_step", before=before_forward_step)

    original_get = getattr(module, "get_forward_backward_func", None)
    if original_get is None:
        return False
    if getattr(original_get, "_hostmem_patched", False):
        return ok

    def get_forward_backward_func(*args, **kwargs):
        fn = original_get(*args, **kwargs)
        if fn is None or getattr(fn, "_hostmem_patched", False):
            return fn

        def wrapped(*fargs, **fkwargs):
            out = fn(*fargs, **fkwargs)
            if _in_window():
                anchor(_step_label("fwdbwd_done"))
            return out

        wrapped._hostmem_patched = True
        wrapped.__wrapped__ = fn
        return wrapped

    get_forward_backward_func._hostmem_patched = True
    module.get_forward_backward_func = get_forward_backward_func
    _state["patched"].add("schedules.get_forward_backward_func")
    return ok


def _patch_optimizer(module):
    ok = True
    for cls_name in ("ChainedOptimizer",):
        cls = getattr(module, cls_name, None)
        if cls is None:
            ok = False
            continue

        def before_step(*args, **kwargs):
            if _in_window():
                anchor(_step_label("optim_begin"))

        def after_step(result, *args, **kwargs):
            if _in_window():
                anchor(_step_label("optim_end"))

        ok = _wrap_function(cls, "step", before=before_step, after=after_step, tag=f"{cls_name}.step") and ok
    return ok


def _patch_offloader(module):
    cls = getattr(module, "ChunkedOptimizerStateOffloader", None)
    if cls is None:
        return False

    def before_step(*args, **kwargs):
        if _in_window():
            anchor(_step_label("offloader_step_begin"))

    def after_step(result, *args, **kwargs):
        if _in_window():
            anchor(_step_label("offloader_step_end"))

    return _wrap_function(cls, "step", before=before_step, after=after_step, tag="ChunkedOptimizerStateOffloader.step")


def _patch_newton_schulz(module):
    def before_ns(*args, **kwargs):
        if not _state["ns_seen"]:
            _state["ns_seen"] = True
            anchor("first_newton_schulz")

    ok = _wrap_function(module, "newton_schulz", before=before_ns, tag="muon_utils.newton_schulz")
    _wrap_function(module, "newton_schulz_tp", before=before_ns, tag="muon_utils.newton_schulz_tp")  # optional
    return ok


def _patch_triton_errors(module):
    cls = getattr(module, "OutOfResources", None)
    if cls is None:
        return False
    original = cls.__init__
    if getattr(original, "_hostmem_patched", False):
        return True

    def __init__(self, *args, __original=original, **kwargs):
        try:
            now = round(time.time() - _T0, 3)
            with _lock:
                _state["oor_count"] += 1
                if _state["oor_first_t"] is None:
                    _state["oor_first_t"] = now
                _state["oor_last_t"] = now
        except Exception:  # pylint: disable=broad-except
            pass
        return __original(self, *args, **kwargs)

    __init__._hostmem_patched = True
    cls.__init__ = __init__
    _state["patched"].add("triton.OutOfResources.__init__")
    return True


def _register_instances(cls, registry_key, tag):
    original = getattr(cls, "__init__", None)
    if original is None:
        return False
    if getattr(original, "_hostmem_patched", False):
        return True

    def __init__(self, *args, __original=original, **kwargs):
        __original(self, *args, **kwargs)
        try:
            _state[registry_key].add(self)
        except Exception:  # pylint: disable=broad-except
            pass

    __init__._hostmem_patched = True
    cls.__init__ = __init__
    _state["patched"].add(tag)
    return True


def _patch_fgo(module):
    """Fine-grained activation offload: register pools / managers / handlers, count offload + reload."""
    ok = True
    pool_cls = getattr(module, "OffloadTensorPool", None)
    ok = (_register_instances(pool_cls, "pools", "OffloadTensorPool.__init__") if pool_cls else False) and ok
    mgr_cls = getattr(module, "PipelineOffloadManager", None)
    if mgr_cls is not None:
        _register_instances(mgr_cls, "managers", "PipelineOffloadManager.__init__")
    handler_cls = getattr(module, "ChunkOffloadHandler", None)
    if handler_cls is None:
        return False
    _register_instances(handler_cls, "handlers", "ChunkOffloadHandler.__init__")

    original_offload = getattr(handler_cls, "offload", None)
    if original_offload is not None and not getattr(original_offload, "_hostmem_patched", False):

        def offload(self, src_tensor, *args, __original=original_offload, **kwargs):
            state = __original(self, src_tensor, *args, **kwargs)
            try:
                view_bytes = int(src_tensor.numel() * src_tensor.element_size())
                host_bytes = 0
                backup = state[1] if isinstance(state, (tuple, list)) and len(state) > 1 else None
                if backup is not None and hasattr(backup, "untyped_storage"):
                    host_bytes = int(backup.untyped_storage().nbytes())
                with _lock:
                    _state["offload_calls"] += 1
                    _state["offload_view_bytes"] += view_bytes
                    _state["offload_host_bytes"] += host_bytes
                    if host_bytes > view_bytes:
                        _state["offload_storage_path_calls"] += 1
            except Exception:  # pylint: disable=broad-except
                pass
            return state

        offload._hostmem_patched = True
        handler_cls.offload = offload
        _state["patched"].add("ChunkOffloadHandler.offload")
    else:
        ok = ok and original_offload is not None

    original_reload = getattr(handler_cls, "reload", None)
    if original_reload is not None and not getattr(original_reload, "_hostmem_patched", False):

        def reload(self, *args, __original=original_reload, **kwargs):
            before = 0
            try:
                import torch  # noqa: PLC0415

                before = torch.cuda.memory_allocated()
            except Exception:  # pylint: disable=broad-except
                pass
            out = __original(self, *args, **kwargs)
            try:
                import torch  # noqa: PLC0415

                delta = torch.cuda.memory_allocated() - before
                out_bytes = int(out.numel() * out.element_size()) if hasattr(out, "numel") else 0
                with _lock:
                    _state["reload_calls"] += 1
                    _state["reload_dev_delta_bytes"] += max(delta, 0)
                    _state["reload_out_bytes"] += out_bytes
            except Exception:  # pylint: disable=broad-except
                pass
            return out

        reload._hostmem_patched = True
        handler_cls.reload = reload
        _state["patched"].add("ChunkOffloadHandler.reload")
    else:
        ok = ok and original_reload is not None
    return ok


def _install_gc_trace():
    import gc  # noqa: PLC0415

    state = {"t0": None}

    def _cb(phase, info):
        if phase == "start":
            state["t0"] = time.perf_counter()
            return
        try:
            with _lock:
                _state["gc_events"] += 1
                _state["gc_collected"] += int(info.get("collected", 0) or 0)
            stack = traceback.extract_stack(sys._getframe(1), limit=14)  # pylint: disable=protected-access
            desc = " < ".join(f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}" for fr in reversed(stack))
            rec = {"type": "gc", "rank": _state["rank"], "step": _state["step"], "t": round(time.time() - _T0, 3),
                   "generation": info.get("generation"), "collected": info.get("collected"),
                   "ms": round((time.perf_counter() - state["t0"]) * 1e3, 1) if state["t0"] else None, "stack": desc}
            _state["gc_last"] = rec
            _write(rec)
            if _state["rank"] == 0:
                print(f"[HOSTMEM] gc rank=0 step={_state['step']} gen={rec['generation']} collected={rec['collected']} "
                      f"{rec['ms']} ms | {desc[:300]}", flush=True)
        except Exception:  # pylint: disable=broad-except
            pass

    gc.callbacks.append(_cb)


_TARGETS = {
    "megatron.training.training": _patch_training,
    "megatron.training.kernel_warmup": _patch_kernel_warmup,
    "megatron.core.pipeline_parallel.schedules": _patch_schedules,
    "megatron.core.optimizer.optimizer": _patch_optimizer,
    "megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload": _patch_offloader,
    "megatron.core.pipeline_parallel.fine_grained_activation_offload": _patch_fgo,
    "emerging_optimizers.orthogonalized_optimizers.muon_utils": _patch_newton_schulz,
    "triton.runtime.errors": _patch_triton_errors,
}
_DONE = set()
_ORIGINAL_IMPORT = None


def _patch_loaded_targets():
    for name, patcher in _TARGETS.items():
        if name in _DONE:
            continue
        module = sys.modules.get(name)
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        if spec is not None and getattr(spec, "_initializing", False):
            continue  # module body still running (we are inside its import): retry on the next import
        try:
            done = bool(patcher(module))
        except Exception:  # pylint: disable=broad-except
            done = True  # do not retry a patcher that raises
        if done:
            _DONE.add(name)


def _install_import_hook():
    global _ORIGINAL_IMPORT  # pylint: disable=global-statement
    if _ORIGINAL_IMPORT is not None:
        return
    _ORIGINAL_IMPORT = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):  # pylint: disable=redefined-builtin
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        if len(_DONE) < len(_TARGETS) and (name.startswith("megatron") or name.startswith("emerging_optimizers")
                                           or name.startswith("triton")):
            _patch_loaded_targets()
        return module

    builtins.__import__ = hooked_import


def _at_exit():
    try:
        anchor("exit", {"patched": sorted(_state["patched"])})
        with _lock:
            top = [
                {"stack": sig, "cum_bytes": value, "live_bytes": _state["pinned_live_by_stack"][sig],
                 "count": _state["pinned_cnt_by_stack"][sig]}
                for sig, value in _state["pinned_cum_by_stack"].most_common(_TOP_STACKS * 2)
            ]
        _write({"type": "pinned_summary", "rank": _state["rank"], "stacks": top,
                "pinned_py_cum": _state["pinned_cum_total"], "pinned_py_live": _state["pinned_live_total"]})
    except Exception:  # pylint: disable=broad-except
        pass


def _main():
    if not _ENABLED:
        return
    if os.environ.get("MCORE_HOSTMEM_PROBE_NO_FACTORY") != "1":
        try:
            _patch_pinned_factories()
        except Exception:  # pylint: disable=broad-except
            pass
    _install_import_hook()
    _patch_loaded_targets()
    try:
        _install_gc_trace()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        threading.Thread(target=_sampler, name="hostmem-probe", daemon=True).start()
    except Exception:  # pylint: disable=broad-except
        pass
    atexit.register(_at_exit)
    anchor("probe_start", {"pid": os.getpid(), "steps": _STEPS, "interval": _INTERVAL,
                           "argv": sys.argv[:3], "out_dir": _out_dir()})


_main()
