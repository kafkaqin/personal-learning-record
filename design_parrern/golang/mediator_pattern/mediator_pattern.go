package main

import "fmt"

type Mediator interface {
	SendMessage(message string, colleague Colleague)
	AddColleague(colleague Colleague)
}

type Colleague interface {
	SetMediator(mediator Mediator)
	GetName() string
	ReceiveMessage(message string)
}

type ChatRoom struct {
	colleagues []Colleague
}

func (chatRoom *ChatRoom) AddColleague(colleague Colleague) {
	chatRoom.colleagues = append(chatRoom.colleagues, colleague)
	colleague.SetMediator(chatRoom)
}

func (chatRoom *ChatRoom) SendMessage(message string, sender Colleague) {
	for _, colleague := range chatRoom.colleagues {
		if colleague.GetName() != sender.GetName() {
			colleague.ReceiveMessage(fmt.Sprintf("[%s] %s", sender.GetName(), message))
		}
	}
}

type ConcreteColleague struct {
	name     string
	mediator Mediator
}

func (cc *ConcreteColleague) SetMediator(mediator Mediator) {
	cc.mediator = mediator
}

func (cc *ConcreteColleague) GetName() string {
	return cc.name
}

func (cc *ConcreteColleague) ReceiveMessage(message string) {
	fmt.Printf("%s received: %s\n", cc.name, message)
}

func (cc *ConcreteColleague) SendMessage(message string) {
	fmt.Printf("%s sent: %s\n", cc.name, message)
	cc.mediator.SendMessage(message, cc)
}

func main() {
	chatRoom := &ChatRoom{
		colleagues: make([]Colleague, 0),
	}

	alice := &ConcreteColleague{
		name: "Alice",
	}
	bob := &ConcreteColleague{
		name: "Bob",
	}
	charlie := &ConcreteColleague{
		name: "Charlie",
	}

	chatRoom.AddColleague(alice)
	chatRoom.AddColleague(bob)
	chatRoom.AddColleague(charlie)

	alice.SendMessage("Hello,everyone")
	bob.SendMessage("Hi,alice")
}
