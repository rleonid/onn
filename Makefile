
default:
	ocamlbuild -package lacaml -I src test.native

onn:
	ocamlbuild -package lacaml -I src onn.cma

clean:
	ocamlbuild -clean
