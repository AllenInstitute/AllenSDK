PROJECTNAME = allensdk
DISTDIR = dist
BUILDDIR = build
export RELEASE=dev$(BUILD_NUMBER)
RELEASEDIR = $(PROJECTNAME)-$(VERSION).$(RELEASE)
EGGINFODIR = $(PROJECTNAME).egg-info
DOCDIR = doc

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME)
	cp -r allensdk setup.py README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz $(PROJECTNAME)
	

distutils_build: clean
	python setup.py build
	
setversion:
	sed -ie 's/'\''[0-9]\+.[0-9]\+.[0-9]\+.dev[0-9]\+'\''/'\''${VERSION}.${RELEASE}'\''/g' allensdk/__init__.py

sdist: distutils_build
	python setup.py sdist

doc: FORCE
	sphinx-apidoc -d 4 -H "Allen SDK" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION).dev$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc
	cp -R doc_template/examples doc
	cp -R doc_template/aibs_sphinx/static/* doc/_static
	cp -R doc_template/aibs_sphinx/templates/* doc/_templates
	sed -ie "s/|version|/${VERSION}.${RELEASE}/g" doc/install.rst
	sed -ie "s/|version|/${VERSION}.${RELEASE}/g" doc/links.rst
	sed -ie "s/\/external_assets/_static\/external_assets/g" doc/_templates/portalHeader.html
	sed -ie "s/\/external_assets/_static\/external_assets/g" doc/_static/external_assets/javascript/portal.js
	cd doc && make html || true
	cp doc_template/.nojekyll doc/_build/html

FORCE:

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
