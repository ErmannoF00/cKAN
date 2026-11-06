.PHONY: build test check benchmark quick-bench trained-bench plot sanitized clean
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j2
test: build
	ctest --test-dir build --output-on-failure
check: build
	python3 tools/train.py --infer build/kan_infer
benchmark: build
	./build/kan_bench --output results/bench_host.csv
quick-bench: build
	./build/kan_bench --quick --output results/bench_host.csv
trained-bench: build
	python3 tools/sweep_trained.py
plot:
	python3 tools/plot_benchmark.py
	python3 tools/summarize_benchmark.py
sanitized:
	cmake -S . -B build-sanitized -DCMAKE_BUILD_TYPE=Debug -DKAN_SANITIZE=ON
	cmake --build build-sanitized -j2
	ctest --test-dir build-sanitized --output-on-failure
clean:
	rm -rf build build-sanitized
