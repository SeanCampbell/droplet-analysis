{ pkgs, ... }: {
    services.docker.enable = true;
    packages = [
        pkgs.python311
        pkgs.python311Packages.pip
        pkgs.python311Packages.ipykernel
    ];
    idx = {
        extensions = [
            "ms-python.python"
            "ms-python.debugpy"
            "ms-toolsai"
            "ms-toolsai.jupyter"
        ];
        workspace = {
            onCreate = {
                create-venv = ''
                    python -m venv .venv
                    source .venv/bin/activate
                    pip install -r deploy/requirements.txt
                '';
                # create-docker = ''
                #     docker build deploy/ -t droplet-analysis
                #     docker run -v ~/droplet-analysis:/app -it droplet-analysis bash
                # '';
            };
        };
    };
}