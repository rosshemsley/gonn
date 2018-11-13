package main

import (
	"fmt"
	"log"
	"os"

	"github.com/rosshemsley/gonn/mnist"
)

func main() {
	labels, err := mnist.LoadLabelsGzipFile("/Users/ross/repos/gonn/data/train-labels-idx1-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to read file: %s", err)
	}
	nRows, _ := labels.Dims()
	log.Printf("Loaded %d labels", nRows)

	imgs, err := mnist.LoadImagesGzipFile("/Users/ross/repos/gonn/data/train-images-idx3-ubyte.gz")
	if err != nil {
		log.Fatalf("Failed to read file: %s", err)
	}

	nRows, _ = imgs.Dims()
	log.Printf("Loaded %d images", nRows)

	for i := 0; i < 5000; i++ {
		label := mnist.LabelValue(labels.RawRowView(i))
		path := fmt.Sprintf("out/img_%d_label_%d.png", i, label)

		out, err := os.Create(path)
		if err != nil {
			log.Fatalf("%s", err)
		}
		mnist.WritePNG(out, imgs.RawRowView(i))

	}
}
