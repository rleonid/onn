
default:
	ocamlbuild -package lacaml -I src test.native

clean:
	ocamlbuild -clean
