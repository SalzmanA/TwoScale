.PHONY: def doc clean copyr

def:
	@PYTHONPATH=${PWD}/doc:${PYTHONPATH} make -s doc
doc: doxy/xml/index.xml copyr
	jupyter-book build .

doxy/xml/index.xml:../cpp/dolfinx/*.h
	doxygen doxy/Doxyfile

copyr:
	V=`grep "project(Twos" ../cpp/CMakeLists.txt`; \
	NDQ=`echo $$V | grep -c '"' `; \
	if [ $$NDQ -gt 0 ]; \
	then  VER=`echo $$V | cut -d '"' -f 2`;\
	else VER=`echo $$V | cut -d ' ' -f 3| sed 's/)//'`;\
	fi; \
	NBD=`echo $$VER |grep -o '\.'|wc -l`; \
	if [ $$NBD -ne 2 ]; \
	then echo "Cmakefile twoscale project version is not parsed correctelly. CHECK"; \
	exit 1; \
	else sed "s/@PROJECT_VERSION@/$$VER/" _templates/copyright.html.in > _templates/copyright.html; fi

clean:
	rm -rf doxy/xml
	rm -rf _build

