package main

import (
	"log"
	"net/http"
	"time"
	"os"
	"errors"
	"strconv"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

var CookieName = "cookie_id"


func HealthHandler(ctx *gin.Context){
	ctx.JSON(http.StatusOK, gin.H{"status" : "Server Alive"})
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
	history, err := GetAllFromDB[FrontHistoryMessageForm]("session_id = ?", session_id, &Messages{})
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
		log.Printf("Неверный формат пришедшего JSON | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
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

// /api/register
func RegisterCustomer(ctx *gin.Context){
	var cust_data CustomerRegisterData
	if err := ctx.ShouldBindJSON(&cust_data); err != nil {
		log.Printf("Неверный формат пришедшего JSON | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusBadRequest, FillErrorForm("Invalid JSON format", http.StatusBadRequest, err))
		return
	}
	salt_size, _ := strconv.Atoi(os.Getenv("SALT_SIZE"))
	bytes, err := bcrypt.GenerateFromPassword([]byte(cust_data.Password), salt_size)
	if err != nil {
		log.Printf("Неудалось захэшировать пароль | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Invalid password hashing", http.StatusInternalServerError, err))
		return
	}
	customer_id := uuid.NewString()

	maxAge := 3600 * 24 * 7
	expires_at := time.Now().Add(time.Duration(maxAge) * time.Second)
	cookie_id := uuid.NewString();
	
	
	newCustomer := Customers{
		CustomerId: customer_id,
		Login:      cust_data.Login,
		Password:   string(bytes),
		IsAdmin:    cust_data.IsAdmin,
		Cookies: []Cookies{
			{
				CookieId: cookie_id,
				CustomerId: customer_id,
				ExpiresAt: expires_at,
			},
		},
	}
	if err := globalDBCon.Create(&newCustomer).Error; err != nil {
		log.Printf("Неудалось создать запись о Пользователе в БД | err: %s", err.Error())
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}	
	ctx.SetCookie(CookieName, cookie_id, 3600 * 24 * 7, "/", "", false, false)
	ctx.Set(CookieName, cookie_id)   
	var response = RegisterResponse {
		Login: cust_data.Login,
		IsAdmin: cust_data.IsAdmin,
	}
	ctx.JSON(http.StatusOK, response)
}

// /api/logout
func LogoutCustomer(ctx *gin.Context){
	ctx.SetCookie(CookieName, "", -1, "/", "", false, true)
	ctx.Status(http.StatusOK)
}

// /api/login/cookie
func LoginCustomerCookie(ctx *gin.Context){
	val, exists := ctx.Get(CookieName)
	if !exists {
		err := errors.New("Cookie_id wasn't exist in context map")
		log.Printf("Неудалось найти cookie_id в хранилище контекста | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Middleware Map Error", http.StatusInternalServerError, err))
		return
	}
	cookie_id, ok := val.(string)

	if !ok {
		err := errors.New("Can't cast cookie_id value to string")
		log.Printf("Неудалось правильно преобразовать cookie_id из контекста | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Middleware Map Error", http.StatusInternalServerError, err))
		return
	}

	cookie_struct, err := GetOneFromDB[Cookies]("cookie_id = ?", cookie_id, &Cookies{})
	if err != nil {
		log.Printf("Неудалось найти Cookie в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.SetCookie(CookieName, "", -1, "/", "", false, true)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}
	customer_data, err := GetOneFromDB[Customers]("customer_id = ?", cookie_struct.CustomerId, &Customers{})
	if err != nil {
		log.Printf("Неудалось найти Customer в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.SetCookie(CookieName, "", -1, "/", "", false, true)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}

	server_response := RegisterResponse{
		Login: customer_data.Login,
		IsAdmin: customer_data.IsAdmin,
	}
	ctx.JSON(http.StatusOK, server_response)
}


// /api/login/manual
func LoginCustomerManual(ctx *gin.Context){
	var cust_data CustomerLoginData
	if err := ctx.ShouldBindJSON(&cust_data); err != nil {
		log.Printf("Неверный формат пришедшего JSON | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusBadRequest, FillErrorForm("Invalid JSON format", http.StatusBadRequest, err))
		return
	}

	customer_record, err := GetOneFromDB[Customers]("login = ?", cust_data.Login, &Customers{})
	if err != nil {
		log.Printf("Неудалось корректно найти пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}

	salt_size, _ := strconv.Atoi(os.Getenv("SALT_SIZE"))
	bytes, err := bcrypt.GenerateFromPassword([]byte(cust_data.Password), salt_size)
	if err != nil {
		log.Printf("Неудалось захэшировать пароль | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Invalid password hashing", http.StatusInternalServerError, err))
		return
	}
	if customer_record.Password != string(bytes){
		log.Printf("Данные не совпадают | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusUnauthorized, FillErrorForm("Invalid Customer Data", http.StatusUnauthorized, err))
		return
	}

	maxAge := 3600 * 24 * 7
	expires_at := time.Now().Add(time.Duration(maxAge) * time.Second)
	cookie_id := uuid.NewString();

	cookie_record := Cookies{
		CookieId: cookie_id,
		CustomerId: customer_record.CustomerId,
		ExpiresAt: expires_at,
	}
		
	if err := globalDBCon.Create(&cookie_record).Error; err != nil {
		log.Printf("Неудалось создать запись о Пользователе в БД | err: %s", err.Error())
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}	
	
	server_response := RegisterResponse{
		Login: customer_record.Login,
		IsAdmin: customer_record.IsAdmin,
	}
	ctx.JSON(http.StatusOK, server_response)

}