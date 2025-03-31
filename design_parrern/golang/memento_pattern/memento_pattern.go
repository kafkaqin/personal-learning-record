package main

import "fmt"

type Memento struct {
	state string
}

type Originator struct {
	state string
}

func (o *Originator) SetState(state string) {
	o.state = state
}
func (o *Originator) GetState() string {
	return o.state
}
func (o *Originator) SaveState() *Memento {
	return &Memento{o.state}
}
func (o *Originator) RestoreState(m *Memento) {
	o.state = m.state
}

type Caretaker struct {
	mementos []*Memento
}

func (c *Caretaker) Add(m *Memento) {
	c.mementos = append(c.mementos, m)
}

func (c *Caretaker) Get(index int) *Memento {
	if index >= len(c.mementos) || index < 0 {
		return nil
	}
	return c.mementos[index]
}
func main() {
	originator := &Originator{
		state: "State1",
	}
	caretaker := &Caretaker{}

	fmt.Println("初始状态", originator.SaveState())
	caretaker.Add(originator.SaveState())

	originator.SetState("State2")
	fmt.Println("修改后的状态", originator.GetState())
	caretaker.Add(originator.SaveState())

	originator.SetState("State3")
	fmt.Println("再次修改后的状态", originator.GetState())

	originator.RestoreState(caretaker.Get(1))
	fmt.Println("恢复状态：", originator.GetState())

	originator.RestoreState(caretaker.Get(0))
	fmt.Println("恢复到最初状态：", originator.GetState())
}
