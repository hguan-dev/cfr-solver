# proxy for deps
DEPS := build/conan_toolchain.cmake

.PHONY: deps
$(DEPS):
	mkdir -p build
	conan install . --output-folder=build --build=missing

.PHONY: build
build: $(DEPS)
	cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug
	cmake --build build

.PHONY: run
run: build
	./build/cfr_solver

.PHONY: clean
clean:
	rm -rf build
	rm -rf CMakeUserPresets.json

.PHONY: test
test: build
	./build/bin/cfr_test

.PHONY: format
format:
	python -m isort model/*.py
	python -m black model/*.py
