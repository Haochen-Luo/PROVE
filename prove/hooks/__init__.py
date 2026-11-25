"""Runtime hooks for Agent Lightning, verl and vLLM. They wrap functions of the installed packages when those
are imported; no package source is edited. ``worker_setup.register_in_worker`` installs all of them."""
