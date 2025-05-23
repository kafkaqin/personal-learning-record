package volume_plugin

// This file exists to force the desired plugin implementations to be linked.
import (
	"k8s.io/component-base/featuregate"
	"k8s.io/utils/exec"

	// Volume plugins
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/cephfs"
	"k8s.io/kubernetes/pkg/volume/configmap"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/downwardapi"
	"k8s.io/kubernetes/pkg/volume/emptydir"
	"k8s.io/kubernetes/pkg/volume/fc"
	"k8s.io/kubernetes/pkg/volume/flexvolume"
	"k8s.io/kubernetes/pkg/volume/git_repo"
	"k8s.io/kubernetes/pkg/volume/hostpath"
	"k8s.io/kubernetes/pkg/volume/iscsi"
	"k8s.io/kubernetes/pkg/volume/local"
	"k8s.io/kubernetes/pkg/volume/nfs"
	"k8s.io/kubernetes/pkg/volume/projected"
	"k8s.io/kubernetes/pkg/volume/secret"

	// Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"
)

// ProbeVolumePlugins collects all volume plugins into an easy to use list.
func ProbeVolumePlugins(featureGate featuregate.FeatureGate) ([]volume.VolumePlugin, error) {
	allPlugins := []volume.VolumePlugin{}

	// The list of plugins to probe is decided by the kubelet binary, not
	// by dynamic linking or other "magic".  Plugins will be analyzed and
	// initialized later.
	//
	// Kubelet does not currently need to configure volume plugins.
	// If/when it does, see kube-controller-manager/app/plugins.go for example of using volume.VolumeConfig
	var err error
	allPlugins, err = appendLegacyProviderVolumes(allPlugins, featureGate)
	if err != nil {
		return allPlugins, err
	}
	allPlugins = append(allPlugins, emptydir.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, git_repo.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, hostpath.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins(volume.VolumeConfig{})...)
	allPlugins = append(allPlugins, secret.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, iscsi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, cephfs.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, downwardapi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, fc.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, configmap.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, projected.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, local.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, csi.ProbeVolumePlugins()...)
	return allPlugins, nil
}

// GetDynamicPluginProber gets the probers of dynamically discoverable plugins
// for kubelet.
// Currently only Flexvolume plugins are dynamically discoverable.
func GetDynamicPluginProber(pluginDir string, runner exec.Interface) volume.DynamicPluginProber {
	return flexvolume.GetDynamicPluginProber(pluginDir, runner)
}
