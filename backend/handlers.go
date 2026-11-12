package main

import (
	"log"
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
// 4.1 POST /api/sessions (Наверное готово)
func CreateSessionHandler(ctx *gin.Context){
	var session Session
	uuid := uuid.NewString()
	created_time := time.Now()
	session.SessionId = uuid
	session.CreatedAt = created_time
	LogStruct("Структура сессии для записи в БД", session)
	err := globalDBCon.Create(&session).Error
	if err != nil {
		log.Printf("Неудалось создать запись о Сессии в БД | err: %s", err.Error())
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}
	var session_response SessionForm
	session_response.CreatedAt = created_time.GoString()
	session_response.SessionId = uuid
	session_response.Status = "active"
	LogStruct("Структура ответа при создании Сессии", session_response)
	ctx.JSON(http.StatusOK, session_response)
}


// 4.2 POST /api/chat 
// Данный handler обрабатывает сообщение от пользователя в конкретном чате записывает его в историю а также отправяет LLM модели ждет ответ и отправялет обратно на front
func UserMessageHandler(ctx *gin.Context) {
	var user_request UserRequest
	if err := ctx.ShouldBindJSON(&user_request); err != nil {
		log.Printf("Неудалось обработать пришедший JSON | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusBadRequest, FillErrorForm("Invalid JSON format", http.StatusBadRequest, err))
		return
	}
	
	LogStruct("Сообщение от пользователя", user_request)

	ai_response, err := GetAIAnwser(user_request)
	if err != nil {
		log.Printf("Неудалось получить ответ от ИИ | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(503, FillErrorForm("AI Response Error", 503, err))
		return
	}
	LogStruct("Ответ ИИ", ai_response)
	err = InsertMessage(user_request)
	if err != nil {
		log.Printf("Неудалось записать в сообщение пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}

	err = InsertMessage(ai_response)
	if err != nil {
		log.Printf("Неудалось записать в ответ ИИ в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}
	ctx.JSON(http.StatusOK, ai_response)
}


// Данный Handler необходим чтобы найти все сообщения Ai и User по session_id для отрисовки пользователю
// 4.3 GET /api/sessions/{session_id}/messages
func ChatHistoryHandler(ctx *gin.Context){
	session_id := ctx.Param("session_id")
	_ = session_id
	var respons UserHistoryForm
	history, err := GetAllFromDB[FrontHistoryMessageForm](session_id, &Messages{})
	if err != nil {
		log.Printf("Неудалось корректно найти историю в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}
	respons.SessionId = session_id
	respons.History = history
	LogStruct("История сессии для отрисовки", respons)
	ctx.JSON(http.StatusOK, respons)

}


// Данный Handler принимает от пользователя форму и записывает ее в БД
// 4.4. POST /api/leads 
func ManagerStatmentHandler(ctx *gin.Context){
	var statment_form ManagerStatementForm
	if err := ctx.ShouldBindJSON(&statment_form); err != nil {
		ctx.JSON(http.StatusBadRequest, FillErrorForm("Invalid JSON format", http.StatusBadRequest, err))
		return
	}
	// дальше запрос сохраняем в БД а также созданную структуру LeadForm !!
	var lead_form LeadForm
	lead_form.CreatedAt = time.Now().GoString()
	lead_form.Status = "new"
	lead_form.LeadId = uuid.NewString()
	ctx.JSON(http.StatusCreated, lead_form)
}