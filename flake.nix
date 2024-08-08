{
  description = "An FHS shell with conda and cuda.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };


  outputs = inputs @ {
    self,
    nixpkgs,
    nixpkgs-unstable,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      imports = [];
      perSystem = {system, ...}: let
        lib = (import nixpkgs-unstable {inherit system;}).lib;
        hostname = builtins.getEnv "hostname";
        notLaptop = (hostname == "lungomare" || hostname == "ooshirosagi");
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = notLaptop;
          };
        };
        customKernel =
          if notLaptop
          then pkgs.zfs_unstable.latestCompatibleLinuxPackages
          else pkgs.linuxPackages_latest;
        installationPath = "/home/ks/.conda";
        minicondaScript = pkgs.stdenv.mkDerivation rec {
          name = "miniconda-${version}";
          version = "24.5.0";
          src = pkgs.fetchurl {
            url = "https://repo.anaconda.com/miniconda/Miniconda3-py311_${version}-0-Linux-x86_64.sh";
            sha256 = "OLIDux8r54tzXrwAFi8p6Oc/zZphntWYBJCnIZPuH1g=";
          };
          unpackPhase = "true";
          installPhase = ''
            mkdir -p $out
            cp $src $out/miniconda.sh
          '';
          fixupPhase = ''
            chmod +x $out/miniconda.sh
          '';
        };
        customConda =
          pkgs.runCommand "conda-install"
          {buildInputs = [pkgs.makeWrapper minicondaScript];}
          ''
            mkdir -p $out/bin
            makeWrapper                            \
              ${minicondaScript}/miniconda.sh      \
              $out/bin/conda-install               \
              --add-flags "-p ${installationPath}" \
              --add-flags "-b"
          '';
        defaultDeps = [
          pkgs.python311
          pkgs.ruff
          pkgs.nodejs
          pkgs.pyright
          pkgs.jq
          customConda
        ];
        cudaDeps = with pkgs; [
          autoconf
          binutils
          curl
          freeglut
          gcc11
          git
          gitRepo
          gnumake
          gnupg
          gperf
          libGLU
          libGL
          libselinux
          m4
          ncurses5
          procps
          stdenv.cc
          unzip
          util-linux
          wget
          xorg.libICE
          xorg.libSM
          xorg.libX11
          xorg.libXext
          xorg.libXi
          xorg.libXmu
          xorg.libXrandr
          xorg.libXrender
          xorg.libXv
          zlib
          customKernel.nvidia_x11
          cudaPackages_12_1.cudatoolkit
          file
        ];
        libInputsDefault = [pkgs.file pkgs.stdenv.cc pkgs.stdenv.cc.cc.lib];
        libInputsCuda = libInputsDefault ++ [customKernel.nvidia_x11];
        libInputPaths = lib.makeLibraryPath libInputsDefault;
        libInputPathsCuda = lib.makeLibraryPath libInputsCuda;

        cudaExports = lib.optionalString notLaptop ''
          export CUDA_PATH="${pkgs.cudaPackages_12_1.cudatoolkit}"
          export EXTRA_LDFLAGS="-L/lib -L${customKernel.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
        '';

      in {
        _module.args = {inherit pkgs;};
        legacyPackages = pkgs;

        devShells = {
          conda =
            (pkgs.buildFHSUserEnv {
              name = "conda";
              targetPkgs = pkgs: defaultDeps ++ cudaDeps ++ libInputsCuda;
              profile = ''
                # conda
                export PATH="${installationPath}/bin:$PATH"
                export NIX_CFLAGS_COMPILE="-I${installationPath}/include"
                export NIX_CFLAGS_LINK="-L${installationPath}lib"
                export FONTCONFIG_FILE=/etc/fonts/fonts.conf
                export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale

                export LD_LIBRARY_PATH=${libInputPathsCuda}:$LD_LIBRARY_PATH

                export CUDA_PATH="${pkgs.cudaPackages_12_1.cudatoolkit}"
                export EXTRA_LDFLAGS="-L/lib -L${customKernel.nvidia_x11}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"

                export UID_DOCKER=$(id -u)
                export GID_DOCKER=$(id -g)
                export TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
                exec fish
                #echo "eval ~/.conda/bin/conda \"shell.fish\" \"hook\" $argv | source"
              '';
            })
            .env;

          conda-no-cuda =
            (pkgs.buildFHSUserEnv {
              name = "conda-no-cuda";
              targetPkgs = pkgs: defaultDeps ++ libInputsDefault;
              profile = ''
                # conda
                export PATH="${installationPath}/bin:$PATH"
                export NIX_CFLAGS_COMPILE="-I${installationPath}/include"
                export NIX_CFLAGS_LINK="-L${installationPath}lib"
                export FONTCONFIG_FILE=/etc/fonts/fonts.conf
                export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale

                export LD_LIBRARY_PATH=${libInputPaths}:$LD_LIBRARY_PATH

                export UID_DOCKER=$(id -u)
                export GID_DOCKER=$(id -g)
                export TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
                exec fish
                #echo "eval ~/.conda/bin/conda \"shell.fish\" \"hook\" $argv | source"
              '';
            })
            .env;
        };
      };
    };
}
