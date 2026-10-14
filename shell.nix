{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ (pkgs.python3.withPackages (p: with p; [ opencv4 pillow torch transformers pytest ])) ];
}
