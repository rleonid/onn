
PACKAGES=lacaml,nonstd

default:
	ocamlbuild -pkgs $(PACKAGES) -I src test.native

onn:
	ocamlbuild -pkgs $(PACKAGES) -I src onn.cma

long:
	ocamlbuild -pkgs $(PACKAGES) -I src long.native

clean:
	ocamlbuild -clean
