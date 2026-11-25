# contrib/os: anti-wedge templates (inert)

These are **templates**, not installed files. No build target, script, package or test in this tree copies, enables or
applies them; they exist so an operator can review a generic, sanitized starting point and install what they choose
by hand. Nothing here writes to `/sys`, `/proc`, a device node or a power-play table.

| File | Kind | What it does if an operator installs it |
| :--- | :--- | :--- |
| `90-vegajet-kfd.rules` | udev rule | Gives the local `render` group read+write on `/dev/kfd` (mode `0660`), so KFD ioctls work without widening the node to world-writable. |
| `99-vegajet-antiwedge.conf` | sysctl drop-in | Keeps a wedged task, watchdog or OOM condition visible in the journal instead of silently panicking the host. |
| `vegajet-antiwedge.service` | systemd unit | Oneshot that journals the running kernel release and `/proc/cmdline` for post-incident triage. It reads only; it never changes GPU state. |

## Rules for using them

1. Read the file first: every value is a starting point, not a recommendation for your host.
2. Copy it to the right place yourself (`/etc/udev/rules.d/`, `/etc/sysctl.d/`, `/etc/systemd/system/`) and reload the
   matching daemon. Nothing in this repository does that for you.
3. Never install these as part of a benchmark run or a GPU job. They are host-preparation material only.
4. They contain no device addresses, no user names and no machine-specific paths. Replace `render` with the local GPU
   access group if it differs.

The same read-only checks are available without installing anything: `vegajet doctor --os` (Linux ≥ 7.0, `amdgpu`
`vm_block_size` / `vm_size` / `noretry`, `/dev/kfd` access) and `contrib/hardware/check_boot_flags.sh`.
