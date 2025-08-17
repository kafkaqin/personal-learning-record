package main

import "fmt"

package main

import (
"fmt"
"k8s.io/apimachinery/pkg/conversion"

"k8s.io/apimachinery/pkg/runtime"
"k8s.io/apimachinery/pkg/runtime/schema"
)

// ================== 1. 类型定义（同前）==================

type TypeMeta struct {
	APIVersion string `json:"apiVersion,omitempty"`
	Kind       string `json:"kind,omitempty"`
}

type ObjectMeta struct {
	Name string `json:"name,omitempty"`
}

type V1BetaDeployment struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`
	Spec       V1BetaDeploymentSpec   `json:"spec,omitempty"`
	Status     V1BetaDeploymentStatus `json:"status,omitempty"`
}

func (v *V1BetaDeployment) GetObjectKind() schema.ObjectKind { return v }
func (v *V1BetaDeployment) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	v.APIVersion, v.Kind = gvk.Version, gvk.Kind
}
func (v *V1BetaDeployment) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(v.APIVersion, v.Kind)
}
func (v *V1BetaDeployment) DeepCopyObject() runtime.Object {
	out := *v
	return &out
}

type V1BetaDeploymentSpec struct {
	Replicas *int32          `json:"replicas"`
	Template PodTemplateSpec `json:"template"`
}
type V1BetaDeploymentStatus struct{}

// --- v1 ---
type V1Deployment struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`
	Spec       V1DeploymentSpec   `json:"spec,omitempty"`
	Status     V1DeploymentStatus `json:"status,omitempty"`
}

func (v *V1Deployment) GetObjectKind() schema.ObjectKind { return v }
func (v *V1Deployment) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	v.APIVersion, v.Kind = gvk.Version, gvk.Kind
}
func (v *V1Deployment) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(v.APIVersion, v.Kind)
}
func (v *V1Deployment) DeepCopyObject() runtime.Object {
	out := *v
	return &out
}

type V1DeploymentSpec struct {
	Replicas        int32           `json:"replicas"`
	Template        PodTemplateSpec `json:"template"`
	MinReadySeconds int32           `json:"minReadySeconds"`
}
type V1DeploymentStatus struct{}

type PodTemplateSpec struct{}

// ================== 2. 创建 Scheme 并注册转换 ==================

func createScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()

	appsv1beta1 := schema.GroupVersion{Group: "apps", Version: "v1beta1"}
	appsv1 := schema.GroupVersion{Group: "apps", Version: "v1"}

	// ✅ 注册类型：GVK -> Go Type
	scheme.AddKnownTypeWithName(appsv1beta1.WithKind("Deployment"), &V1BetaDeployment{})
	scheme.AddKnownTypeWithName(appsv1.WithKind("Deployment"), &V1Deployment{})

	// ✅ 使用 scheme.AddConversionFunc（正确方式）
	scheme.AddConversionFunc((*V1BetaDeployment)(nil), (*V1Deployment)(nil),
		func(in interface{}, out interface{}, scope conversion.Scope) error {
			inObj := in.(*V1BetaDeployment)
			outObj := out.(*V1Deployment)

			outObj.ObjectMeta = inObj.ObjectMeta

			// ✅ 设置目标 GVK
			outObj.SetGroupVersionKind(schema.GroupVersionKind{
				Group:   "apps",
				Version: "v1",
				Kind:    "Deployment",
			})

			if inObj.Spec.Replicas != nil {
				outObj.Spec.Replicas = *inObj.Spec.Replicas
			} else {
				outObj.Spec.Replicas = 1
			}
			outObj.Spec.Template = inObj.Spec.Template
			outObj.Spec.MinReadySeconds = 10

			return nil
		})

	return scheme
}

// ================== 3. 主函数 ==================

func main() {
	scheme := createScheme()

	// ✅ 获取转换器
	//convertor := scheme

	// 打印注册的类型
	fmt.Println("🔍 Registered Types:")
	for gvk := range scheme.AllKnownTypes() {
		fmt.Printf("  %s\n", gvk)
	}

	// 创建源对象
	v1beta1Deploy := &V1BetaDeployment{
		TypeMeta: TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: ObjectMeta{
			Name: "demo-deploy",
		},
		Spec: V1BetaDeploymentSpec{
			Replicas: func() *int32 { i := int32(3); return &i }(),
		},
	}

	// ✅ 使用 scheme.Convert 转换
	outObj, err := scheme.ConvertToVersion(v1beta1Deploy, schema.GroupVersion{Group: "apps", Version: "v1"})
	if err != nil {
		fmt.Printf("❌ 转换失败: %v\n", err)
		return
	}

	v1Deploy := outObj.(*V1Deployment)

	fmt.Printf("✅ 转换成功！\n")
	fmt.Printf("Kind: %s\n", v1Deploy.Kind)
	fmt.Printf("APIVersion: %s\n", v1Deploy.APIVersion)
	fmt.Printf("Name: %s\n", v1Deploy.Name)
	fmt.Printf("Replicas: %d\n", v1Deploy.Spec.Replicas)
	fmt.Printf("MinReadySeconds: %d\n", v1Deploy.Spec.MinReadySeconds)
}


