{ pkgs, ... }: {
    services.docker.enable = true;
    packages = [
        pkgs.python311
        pkgs.python311Packages.pip
    ];
    idx = {
        extensions = [
            "ms-python.python"
            "ms-python.debugpy"
        ];
    };
}
