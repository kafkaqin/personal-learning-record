package main

import "fmt"

type Device interface {
	IsEnabled() bool
	Enable()
	Disable()
	GetVolume() int
	SetVolume(volume int)
	GetChannel() int
	SetChannel(channel int)
}

type TV struct {
	enabled bool
	volume  int
	channel int
}

func (t *TV) IsEnabled() bool {
	return t.enabled
}

func (t *TV) Enable() {
	t.enabled = true
	fmt.Println("TV is enabled")
}

func (t *TV) Disable() {
	t.enabled = false
	fmt.Println("TV is disabled")
}

func (t *TV) GetVolume() int {
	return t.volume
}

func (t *TV) SetVolume(volume int) {
	t.volume = volume
	fmt.Printf("TV volume set to %d\n", volume)
}

func (t *TV) GetChannel() int {
	return t.channel
}

func (t *TV) SetChannel(channel int) {
	t.channel = channel
	fmt.Printf("TV channel set to %d\n", channel)
}

type Remote interface {
	TogglePower()
	VolumeUp()
	VolumeDown()
	ChannelUp()
	ChannelDown()
}

type BasicRemote struct {
	device Device
}

func (br *BasicRemote) TogglePower() {
	if br.device.IsEnabled() {
		br.device.Enable()
	} else {
		br.device.Disable()
	}
}

func (br *BasicRemote) VolumeUp() {
	br.device.SetVolume(br.device.GetVolume() + 10)
}

func (br *BasicRemote) VolumeDown() {
	br.device.SetVolume(br.device.GetVolume() - 10)
}

func (br *BasicRemote) ChannelUp() {
	br.device.SetChannel(br.device.GetChannel() + 1)
}

func (br *BasicRemote) ChannelDown() {
	br.device.SetChannel(br.device.GetChannel() - 1)
}

type ExtendedRemote struct {
	BasicRemote
}

func (e *ExtendedRemote) Mute() {
	e.device.SetVolume(0)
	fmt.Println("Device muted")
}

func main() {
	tv := &TV{
		volume:  50,
		enabled: false,
		channel: 1,
	}
	basicRemote := &BasicRemote{
		device: tv,
	}
	fmt.Println("Using Basic Remote:")
	basicRemote.TogglePower()
	basicRemote.VolumeUp()
	basicRemote.VolumeDown()
	basicRemote.ChannelUp()

	extendedRemote := &ExtendedRemote{
		BasicRemote: BasicRemote{
			device: tv,
		},
	}
	extendedRemote.Mute()
	extendedRemote.TogglePower()
}
