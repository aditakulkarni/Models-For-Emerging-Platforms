package main

//For reduce channel
type keyValue struct {
	key string
	value int
	functor ReduceFunc
}

type mapReduce struct {
	sharedMap map[string]int
	askchannel chan string
	returnaskchannel chan int
	addchannel chan string
	reducechannel chan keyValue
	returnreducechannel chan keyValue
	stopchannel chan int
}

//Find word that has appeared least frequently so far
func min_word(w1 string, c1 int, w2 string, c2 int) (string, int) {
	if c1 < c2 {
        return w1, c1
    }
    return w2, c2
}

func (m *mapReduce) AddWord(word string) {
	//Send word to returnaskchannel which is used by Listen() for incrementing count
	m.addchannel <- word
}

func (m *mapReduce) Listen() {
	var ok bool
	var val int	
	for {
		select {

			//Perform AddWord function
			case v1 := <-m.addchannel:
				if val, ok = m.sharedMap[v1]; ok {
					m.sharedMap[v1]++
				}
				if !ok {
					m.sharedMap[v1] = 1
				}

			//Perform GetCount function
			case v2 := <-m.askchannel:
				if val, ok = m.sharedMap[v2]; ok {
					m.returnaskchannel <- val
				}
				if !ok {
					m.returnaskchannel <- 0
				}

			//Perform Stop function
			case <-m.stopchannel:
				break

			//Perform Reduce function
			case currentResult := <- m.reducechannel:

				currentResult.value = 1<<63 - 1
				function := currentResult.functor

				for k, v := range m.sharedMap {
					currentResult.key, currentResult.value = function(currentResult.key, currentResult.value, k, v)
				}

				m.returnreducechannel <- currentResult
		}
	}
}

func (m *mapReduce) Stop() {
	//Send 1 to stopchannel which is used by Listen() for stopping the program
	m.stopchannel <- 1
}

func (m *mapReduce) Reduce(functor ReduceFunc, accum_str string, accum_int int) (string, int) {
	var passtochannel keyValue
	passtochannel.key = accum_str
	passtochannel.value = accum_int
	passtochannel.functor = min_word

	//Send accum_str to reducechannel which is used by Listen() for reduction
	m.reducechannel <- passtochannel

	//After reduction, get the result from returnreducechannel sent by Listen()
	result := <- m.returnreducechannel
	return result.key, result.value
}

func (m *mapReduce) GetCount(word string) int {

	//Send word to returnaskchannel which is used by Listen() for acquiring count
	m.askchannel <- word

	//Get the result from returnreducechannel sent by Listen()
	val := <-m.returnaskchannel
	return val
}

func NewChannelMap() *mapReduce {
	return &mapReduce {
		sharedMap : make(map[string]int),
		addchannel : make(chan string, ADD_BUFFER_SIZE),
		askchannel : make(chan string, ASK_BUFFER_SIZE),
		returnaskchannel : make(chan int, 1),
		reducechannel : make(chan keyValue, 1),
		returnreducechannel : make(chan keyValue, 1),
		stopchannel : make(chan int, 1),
	}
}
