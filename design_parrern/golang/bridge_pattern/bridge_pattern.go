package main

/**
桥接模式（Bridge Pattern）是一种结构型设计模式，它的核心思想是将抽象部分与实现部分分离，
使它们可以独立变化。这有助于减少类的数量，并提高代码的可扩展性。
适用场景
系统需要在多个维度上扩展，而这些维度可能会相互组合（如设备类型+操作系统）。

避免继承层次过深，如果一个类的功能需要多个维度的扩展，使用继承可能会导致类的数量爆炸。

希望解耦抽象和具体实现，使它们能够独立变化。
*/
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

	//
	red := &Red{}
	blue := &Blue{}
	circle := &Circle{
		color: red,
	}

	rectangle := Rectangle{
		color: blue,
	}
	circle.Draw()
	rectangle.Draw()

}

type Color interface {
	Fill() string
}

type Red struct {
}

func (r *Red) Fill() string {
	return "red"
}

type Blue struct{}

func (b *Blue) Fill() string {
	return "blue"
}

type Shape interface {
	Draw()
}

type Circle struct {
	color Color
}

func (c *Circle) Draw() {
	fmt.Println("Circle Draw:", c.color)
}

type Rectangle struct {
	color Color
}

func (r *Rectangle) Draw() {
	fmt.Println("Rectangle Draw:", r.color)
}
