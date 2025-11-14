.PHONY: all build clean plot

all: build

build:
	bash cpp/build.sh

plot:
	python py/plot.py --speedup results/execution_time.csv --out results/plots/execution_time.png
	python py/plot.py --history results/serial_history.txt --out results/plots/serial_plot.png

clean:
	rm -rf results/*
	rm -rf cpp/bin/*
