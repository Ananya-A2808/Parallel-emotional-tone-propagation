.PHONY: all build clean test run-perf plot

all: build

build:
	bash cpp/build.sh

test:
	python -m pytest -q

run-perf:
	bash cpp/run_perf.sh

plot:
	python py/plot.py --speedup results/speedup.csv --out results/plots/speedup.png
	python py/plot.py --history results/serial_history.txt --out results/plots/serial_plot.png

clean:
	rm -rf results/*
	rm -rf cpp/bin/*
