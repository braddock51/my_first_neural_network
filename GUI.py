from tkinter import *
from tkinter import filedialog
from network import NeuralNetwork
import csv
import ast
 

# Funcions ====================================================================================

def create_net():
    create_net.net = NeuralNetwork(structure_input.get(), float(lr_input.get()), int(iterations_input.get()))
    button_train['state'] = NORMAL
    button_query['state'] = NORMAL
    
def create_training_window(net):
    train_window = Toplevel()
    train_window.title("Train your network")
    train_window.minsize(500, 400)

    input_path = filedialog.askopenfilename(initialdir="./", title="Enter the training data", filetypes=(("Csv Files", "*.csv"),))
    
    if input_path == () or input_path == "":
        train_window.destroy()
    else:
        data_file = open(input_path, 'r')
        data_reader = csv.reader(data_file, delimiter=",")
        data = [list( map(int, row)) for row in data_reader]
        
        epochs_label = Label(train_window, text="Enter epochs number").pack()
        epochs_input = Entry(train_window, width=5)
        epochs_input.pack()

        train_button = Button(train_window, text="Train", command=lambda: train_net(net, data, epochs_input.get(), train_window))
        train_button.pack()

def create_query_window(net):
    query_window = Toplevel()
    query_window.title("Query your network")
    query_window.minsize(500,400)

    inputs_label = Label(query_window, text="Enter query inputs\n(In the form of x x x...)").pack()
    inputs = Entry(query_window, width=5)
    inputs.pack()

    open_weights.weights = None
    input_weights_label = Label(query_window, text="If you want to use your own weights set, enter the data").pack()
    input_wieghts_button = Button(query_window, text="Open", command=lambda: open_weights())
    input_wieghts_button.pack()
    
    query_button = Button(query_window, text="Query", command=lambda: query_net(net, inputs.get(), query_window, open_weights.weights))
    query_button.pack()



    

def train_net(net, data, epochs, train_window):
    index = int(structure_input.get().split("-")[0])
    for e in range(int(epochs)):
        for row in data:
            input = row[:index]
            target = row[index:]
            net.train(input, target)
    
    filename_label = Label(train_window, text="Enter the filename").pack()
    filename_input = Entry(train_window)
    filename_input.pack()

    save_weights_button = Button(train_window, text="Save weights", command=lambda: save_weights(net, filename_input.get()))
    save_weights_button.pack()

def query_net(net, inputs, query_window, weights=None):
    inputs_list = list( map(int, inputs.split(" ")))
    
    output = net.query(inputs_list, weights)
    output = [list( map(round, i)) for i in output]
    
    query_output = Label(query_window, text=f"The answer is: {output}").pack()

def save_weights(net, filename):
    file = open(f"{filename}.txt", 'w')
    file.write(str(net.weights))
    file.close()
    
def open_weights():
    weights_path = filedialog.askopenfilename(initialdir="./", title="Enter the weights data", filetypes=(("txt Files", "*.txt"),))
    weights_file = open(weights_path, 'r')
    weights = weights_file.read()
    open_weights.weights = ast.literal_eval(weights)
    
    

# App==========================================================================================

root = Tk()
root.title("Neural network")
root.minsize(500, 400)

strucutre_label = Label(root, text="Enter structure of the network\n(In the form of x-x-x...)")
strucutre_label.pack()
structure_input = Entry(root)
structure_input.pack()

lr_label = Label(root, text="Enter learning rate\n(Decimal number)").pack()
lr_input = Entry(root, width=3)
lr_input.pack()

iterations_label = Label(root, text="Enter number of iterations\n(Integer number)").pack()
iterations_input = Entry(root, width=6)
iterations_input.pack()

create_button = Button(root, text="Create", command=create_net)
create_button.pack()

train_label = Label(root, text="Open the network training window").pack()
button_train = Button(root, text="Open", command=lambda: create_training_window(create_net.net), state=DISABLED)
button_train.pack()

query_label = Label(root, text="Open the network query window").pack()
button_query = Button(root, text="Open", command=lambda: create_query_window(create_net.net), state=DISABLED)
button_query.pack()

button_quit = Button(root, text="Exit Program", command=root.quit)
button_quit.pack()
root.mainloop()
#inputs
#width zeby zmienic wielkosc, get() zwraca value
#buttons
#state=DISABLED - wyłącza przycisk
#padx, pady zmienia rozmiar
# zeby dodac event do przycisku - command=nazwa_funkcji
#fg zmienia kolor textu, bg - zmienia kolor tła
#columnspan=int łączy zajmowane kolumny w jedną

