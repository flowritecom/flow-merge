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
      perSystem = {system, ...} @ args: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        installationPath = "/home/ks/.conda";
        minicondaScript = pkgs.stdenv.mkDerivation rec {
          name = "miniconda-${version}";
          version = "24.3.0";
          src = pkgs.fetchurl {
            url = "https://repo.anaconda.com/miniconda/Miniconda3-py311_${version}-0-Linux-x86_64.sh";
            sha256 = "sha256-Tajd5p7KDZvDFCA0miBIUb+ioch664f+DAVRd5ftqsQ=";
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
          pkgs.nodePackages.pyright
          pkgs.jq
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
          zfsUnstable.latestCompatibleLinuxPackages.nvidia_x11_beta
          cudaPackages_12_1.cudatoolkit
          customConda
          file
          ocrmypdf
          tesseract
          ghostscript
        ];

        libInputs = with pkgs; [
          zfsUnstable.latestCompatibleLinuxPackages.nvidia_x11_beta
          file
          ocrmypdf
          tesseract
          ghostscript
          stdenv.cc
          stdenv.cc.cc.lib
        ];
      in {
        _module.args = {inherit pkgs;};
        legacyPackages = pkgs;
        devShells = with pkgs; {
          conda =
            (pkgs.buildFHSUserEnv rec {
              name = "conda";
              targetPkgs = pkgs: (
                with pkgs; defaultDeps ++ cudaDeps
              );
              profile = ''
                # conda
                export PATH="${installationPath}/bin:$PATH"
                export NIX_CFLAGS_COMPILE="-I${installationPath}/include"
                export NIX_CFLAGS_LINK="-L${installationPath}lib"
                export FONTCONFIG_FILE=/etc/fonts/fonts.conf
                export QTCOMPOSE=${pkgs.xorg.libX11}/share/X11/locale

                # cuda
                export CUDA_PATH="${pkgs.cudaPackages_12_1.cudatoolkit}"
                export EXTRA_LDFLAGS="-L/lib -L${pkgs.zfsUnstable.latestCompatibleLinuxPackages.nvidia_x11_beta}/lib"
                export EXTRA_CCFLAGS="-I/usr/include"
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libInputs}"
                export CUDA_VISIBLE_DEVICES="0,1,2"

                # python
                export TORCH_USE_CUDA_DSA="1"
                export TORCH_DEVICE="cuda"
                export TESSDATA_PREFIX="${pkgs.tesseract}/share/tessdata"
                export INFERENCE_RAM="22"
                export DEFAULT_LANG="Finnish"
                export NUM_DEVICES="3"

                export UID_DOCKER=$(id -u)
                export GID_DOCKER=$(id -g)
                export TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
                exec fish
                #eval ~/.conda/bin/conda "shell.fish" "hook" $argv | source
              '';
            })
            .env;
        };
      };
    };
}
