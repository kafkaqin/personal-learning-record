package main

import (
	"fmt"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func main() {
	scheme := runtime.NewScheme()
	_ = appsv1.AddToScheme(scheme)
	gvk := schema.GroupVersionKind{
		Group:   "apps",
		Kind:    "Deployment",
		Version: "v1",
	}
	fmt.Println(gvk)
	obj, err := scheme.New(gvk)
	if err != nil {
		panic(err)
	}
	fmt.Printf("====%T \n", obj)

	deployment := &appsv1.Deployment{}
	gvk2, _, err := scheme.ObjectKinds(deployment)
	if err != nil {
		panic(err)
	}
	fmt.Printf("GVK: %#v\n", gvk2[0])
}
