.PHONY: build clean
build:
	tinystatic -output docs
serve:
	cd docs && python3 -m http.server --bind 127.0.0.1
clean:
	rm -r docs