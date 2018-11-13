package main

import (
	"fmt"
	"os"

	"github.com/rosshemsley/gonn/examples/mnist"
	kingpin "gopkg.in/alecthomas/kingpin.v2"
)

var (
	exampleName = kingpin.Arg("example", "Name of example to run.").Required().String()

	examples = map[string]func(){
		"mnist": mnist.Run,
	}
)

func main() {
	kingpin.Parse()

	run, ok := examples[*exampleName]
	if !ok {
		fmt.Fprintf(os.Stderr, "Example '%s' not found, available examples are:\n", *exampleName)
		for k := range examples {
			fmt.Fprintf(os.Stderr, "  %s\n", k)
		}
		return
	}

	run()
}
