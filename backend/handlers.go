package main 

import (
	"net/http"
	"time"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func Hello(ctx *gin.Context){
	wr := ctx.Writer
	wr.Write([]byte("Hello"))
}

// Данный Handler возвращает uuid нового чата а также 
// дополнительные данные для записи в БД 
// Необходимо ли user_id проверять при создании нового чата ? 
// 4.1 POST /api/sessions
func CreateSessionHandler(ctx *gin.Context){
	// user_id := ctx.Query("user_id")
	// if user_id == ""{
	// 	ctx.JSON(400, gin.H{"error": "Параметр user_id должен быть"})
	// 	return
	// }

	var session Session
	uuid := uuid.NewString()
	created_time := time.Now()

	session.SessionId = uuid
	session.CreatedAt = created_time
	globalDBCon.Create(&session)

	var session_info SessionForm
	session_info.created_at = created_time.GoString()
	session_info.session_id = uuid
	session_info.status = "active"
	ctx.JSON(http.StatusOK, session_info)
	// Также необходимо записать в БД
}


// 4.2 POST /api/chat 
// Данный handler обрабатывает сообщение от пользователя в конкретном чате записывает его в историю а также отправяет LLM модели ждет ответ и отправялет обратно на front
func UserMessageHandler(ctx *gin.Context) {
	var user_request UserRequest
	if err := ctx.ShouldBindJSON(&user_request); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Неверный формат JSON"})
		return
	}
	ai_response, err := GetAIAnwser(user_request)
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Неудалось получить ответ от ИИ"})
	}
	// тут необходимо записать в БД и отправка в Python а также получение ответа, но пока хз каков там формат, потому то что в примере != итог так что пока 200 ответ
	ctx.JSON(http.StatusOK, ai_response)
}


// Данный Handler необходим чтобы найти все сообщения Ai и USer по session_id для отрисовки пользователю
// 4.3 GET /api/sessions/{session_id}/messages 
func ChatHistoryHandler(ctx *gin.Context){
	session_id := ctx.Param("session_id")
	_ = session_id
	// тут дальше поиск из GORM по session_id 
	// а после он вернет уже все в виде одного json и все четко
	ctx.JSON(http.StatusOK, gin.H{"message" : "ok"})

}


// Данный Handler принимает от пользователя форму и записывает ее в БД
// 4.4. POST /api/leads 
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