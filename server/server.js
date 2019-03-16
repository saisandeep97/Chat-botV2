const path = require("path");
const http = require("http");
const express = require("express");
const socketIO = require("socket.io");
const request = require("request");

const {
  generateMessage,
  generateLocationMessage
} = require("./utils/message");
const {
  isRealString
} = require("./utils/validation");
const {
  Users
} = require("./utils/users");

const publicPath = path.join(__dirname, "../public");
const port = process.env.PORT || 3000;
var app = express();
var server = http.createServer(app);
var io = socketIO(server);
var users = new Users();

app.use(express.static(publicPath));

io.on("connection", socket => {
  console.log("New user connected");

  socket.on("join", (params, callback) => {
    if (!isRealString(params.name) || !isRealString(params.room)) {
      return callback("Name and room name are required.");
    }

    socket.join(params.room);
    users.removeUser(socket.id);
    users.addUser(socket.id, params.name, params.room);

    io.to(params.room).emit("updateUserList", users.getUserList(params.room));
    socket.emit(
      "newMessage",
      generateMessage(
        "Admin",
        'Say Hi to gobot with "#gobot Hi" as your message'
      )
    );
    socket.broadcast
      .to(params.room)
      .emit(
        "newMessage",
        generateMessage("Admin", `${params.name} has joined.`)
      );
    callback();
  });

  socket.on("createMessage", (message, callback) => {
    var user = users.getUser(socket.id);

    if (user && isRealString(message.text)) {
      io.to(user.room).emit(
        "newMessage",
        generateMessage(user.name, message.text)
      );
    }
    if (message.text.startsWith("#gobot")) {
      request.post(
        "http://localhost:5000/chatbot", {
          json: {
            message: message.text
          }
        },
        function (error, response, body) {
          if (!error) {
            io.to(user.room).emit(
              "newMessage",
              generateMessage("#gobot", body.text)
            );
          }
        }
      );
    }
    callback();
  });

  socket.on("createLocationMessage", coords => {
    var user = users.getUser(socket.id);

    if (user) {
      io.to(user.room).emit(
        "newLocationMessage",
        (user.name, coords.latitude, coords.longitude)
      );
    }
  });

  socket.on("disconnect", () => {
    var user = users.removeUser(socket.id);

    if (user) {
      io.to(user.room).emit("updateUserList", users.getUserList(user.room));
      io.to(user.room).emit(
        "newMessage",
        generateMessage("Admin", `${user.name} has left.`)
      );
    }
  });
});

server.listen(port, () => {
  console.log(`Server is up on ${port}`);
});