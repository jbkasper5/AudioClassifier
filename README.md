# AudioClassifier
Neural network based audio classifier used to recognize a small set of words to drive a simple maze game. The data was collected using the UI from the `DataCapture` folder, and the model was trained on the resulting set of data. The model was then used after training in real time to guide the game.  

During the game, the player can choose an available direction by hitting space to initiate the recording, where their recording (a spoken form of "up", "down", "left", or "right") will be converted into a spectral image and later into the model (CNN classifier architecture). The model's output determines what it thinks the player said and will move the player accordingly. The game will then tell the user if they've successfully navigated the maze.

Each randomly generated maze is tested to see if it can be completed before it is given to the user using the breadth first search algorithm, and impossible mazes are regenerated until a completable maze is found. 
