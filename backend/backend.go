package main

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func Hello(ctx *gin.Context){
	wr := ctx.Writer
	wr.Write([]byte("Hello"))
}








type SessionForm struct{
	session_id string `json:"session_id"`
	status string `json:"status"`
	created_at string `json:created_at`
}

// Данный Handler возвращает uuid нового чата а также 
// дополнительные данные для записи в БД 
// Необходимо ли user_id проверять при создании нового чата ? 
func CreateSession(ctx *gin.Context){
	// user_id := ctx.Query("user_id")
	// if user_id == ""{
	// 	ctx.JSON(400, gin.H{"error": "Параметр user_id должен быть"})
	// 	return
	// }
	var session_info SessionForm
	session_info.created_at = time.Now().GoString()
	session_info.session_id = uuid.NewString()
	session_info.status = "active"
	ctx.JSON(http.StatusOK, &session_info)
	// Также необходимо записать в БД
}

// Структура для /api/chat с сообщением пользователя
type UserRequest struct{
	session_id string `json:"session_id"`
	request_id string `json:request_id""`
	message string `json:"message"`
}
// Данный handler обрабатывает сообщение от пользователя в конкретном чате записывает его в историю а также отправяет LLM модели ждет ответ и отправялет обратно на front
func UserMessageHandler(ctx *gin.Context) {
	var user_request UserRequest
	if err := ctx.ShouldBindJSON(&user_request); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Неверный формат JSON"})
		return
	}
	// тут необходимо записать в БД и отправка в Python а также получение ответа, но пока хз каков там формат, потому то что в примере != итог так что пока 200 ответ
	ctx.Status(http.StatusOK)
}


// Данный Handler необходим чтобы найти все сообщения Ai и USer по session_id для отрисовки пользователю
func ChatHistoryHandler(ctx *gin.Context){
	session_id := ctx.Param("session_id")
	_ = session_id
	// тут дальше поиск из GORM по session_id 
	// а после он вернет уже все в виде одного json и все четко
	ctx.JSON(http.StatusOK, gin.H{"message" : "ok"})

}

type UserContactData struct {
	name string `json"name"`
	company string `json:"company"`
	phone string `json:"phone"`
	email string `json:"email"`
}

type ManagerStatementForm struct {
	session_id string `json:"session_id"`
	submission_id string `json:"submission_id"`
	contact UserContactData `json:"contact"`
	consent bool `json:consent`
}

type LeadForm struct {
	lead_id string `json:"lead_id"`
	status string `json:"status"`
	created_at string `json:"created_at"`
}

// Данный Handler принимает от пользователя форму и записывает ее в БД
func ManagerStatmentHandler(ctx *gin.Context){
	var statment_form ManagerStatementForm
	if err := ctx.ShouldBindJSON(&statment_form); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Неверный формат JSON"})
		return
	}
	// дальше запрос сохраняем в БД а также созданную структуру LeadForm !!
	var lead_form LeadForm
	lead_form.created_at = time.Now().GoString()
	lead_form.status = "new"
	lead_form.lead_id = uuid.NewString()
	ctx.JSON(http.StatusCreated, lead_form)
}

func main(){
	// Для работы с БД будет исользоваться GORM из-за удобства автоматически создавать таблицы и записывать туда данные создавая их из type struct
	r := gin.Default()
	// Политика CORS чтобы доверять грубо говоря ip, пока что localhost можно поменять позже, пока не понятно где находится front в будущем
	config := cors.Config {
		AllowOrigins: []string{"http://localhost:3000"},
		AllowMethods: []string{"POST", "GET", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-type", "Authorization"},
		AllowCredentials: true,
		MaxAge: 12 * time.Hour,
	}
	r.Use(cors.New(config))

	// позже настрою Cookie 
	r.GET("api/hello", Hello)
	r.GET("api/sessons/:session_id/messages")


	r.POST("api/sessions", CreateSession)
	r.POST("api/chat", UserMessageHandler)
	r.POST("api/leads", ManagerStatmentHandler)

	fmt.Print("The server was opened at http://localhost:8080")
	r.Run(":8080")
}