# # one step
# input_tensor = letter_to_tensor('A')
# hidden_tensor = rnn.init_hidden_layer()
# output, hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(hidden.size())

# # whole sequence/name
# input_tensor = line_to_tensor('Albert')
# hidden_tensor = rnn.init_hidden_layer()

# output, hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.size())
# print(hidden.size())

# print(category_from_output(output))
