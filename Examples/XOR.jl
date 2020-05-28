# A Simple XOR example of SVD-DNN
# It won't show the speed improvements since the model is too small

include("../SVDNN.jl")
include("../SVDNN_utils.jl")

using Flux: onehotbatch, onecold, @epochs
using Flux.Data: DataLoader
using Flux.Optimise

println("Finished loading packages")

# inputs for XOR
# each column contains the vector of input (A and B)
# 4 columns = 4 inputs
X = [0 0 1 1; 0 1 0 1]
X = Array{Float32, 2}(X)

# corresponding outputs (to each columns)
Y = [0, 1, 1, 0]

data = DataLoader(X, Y)

# Model

num_features = 2  # size of each input vector
num_hidden_layer_neurons = 20  # Number of neurons in hidden layer
learning_rate = 0.01

# build model
model = Chain(
              Dense(num_features, num_hidden_layer_neurons, relu),
              Dense(num_hidden_layer_neurons, num_hidden_layer_neurons, relu),
              Dense(num_hidden_layer_neurons, 1, σ)
             )

# store parameters of model
ps = params(model)
println("Model has $(sum(length, ps)) parameters")

# optimizer, gradient descent
opt = Descent(learning_rate)
opt2 = Descent(learning_rate)

# loss function
loss(x, y, model) = Flux.binarycrossentropy(model(x)[1], y[1])

# arrays for plotting graphs
costs = []
accuracies = []
test_accuracies = []

# required functions
function cost_function(data, loss, model)
    losses = []
    for (x, y) in data
        push!(losses, loss(x, y, model))
    end
    avg_loss = sum(losses)/length(losses)
    return avg_loss
end

function accuracy(data, model)
    corrects = []
    for (x, y) in data
        pred = model(x)[1]
        pred = pred >= 0.5 ? 1 : 0
        push!(corrects, pred == y[1])
    end
    return sum(corrects)/length(corrects)
end

# should be tuned in real models
l_values = [1, 5, nothing]

@timeit to "Full training SVD" train_epochs_svd!(model, l_values, loss, data, data, opt, opt2, 800)

println("XOR of 0 and 0 is $(round(Int, model([0; 0])[1]))")
println("XOR of 0 and 1 is $(round(Int, model([0; 1])[1]))")
println("XOR of 1 and 0 is $(round(Int, model([1; 0])[1]))")
println("XOR of 1 and 1 is $(round(Int, model([1; 1])[1]))")

