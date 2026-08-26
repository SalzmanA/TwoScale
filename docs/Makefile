.PHONY: def doc clean

def:
	@PYTHONPATH=${PWD}/doc:${PYTHONPATH} make -s doc
doc: doxy/xml/index.xml
	jupyter-book build .

doxy/xml/index.xml:../cpp/dolfinx/*.h
	doxygen doxy/Doxyfile


clean:
	rm -rf doxy/xml
	rm -rf _build

