package main

import (
	"errors"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
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
	customer_id, err := GetCustomerIdFromDB(ctx)
	if err != nil {
		log.Printf("Ошибка при работе с Cookie | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}
	var session Session
	uuid := uuid.NewString()
	created_time := time.Now()
	session.SessionId = uuid
	session.CreatedAt = created_time
	session.CustomerId = customer_id
	LogStruct("Структура сессии для записи в БД", session)
	err = globalDBCon.Create(&session).Error
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
		log.Printf("Неудалось записать в ответ ИИ в Messages | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}

	err = UpdateCustomerProfile(ai_response)
		if err != nil {
		log.Printf("Неудалось обновить / записать информацию в Profiles| Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}

	err = UpdateSessionInfo(ai_response)
		if err != nil {
		log.Printf("Неудалось обновить информацию в Session | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}

	ai_message_response  := FrontHistoryMessageForm{
		Role: "ai",
		Content: ai_response.Answer,
		Action: ai_response.Action,
		RecProducts: ai_response.RecProducts,
		CreatedAt: time.Now().GoString(),
	}

	LogStruct("Структура ответа ИИ на Front", ai_message_response)
	ctx.JSON(http.StatusOK, ai_message_response)
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
	LogStruct("Данные для заявки с Front", statment_form)
	lead_id := uuid.NewString()
	created_at := time.Now()

	rec_products, err := GetLastRecProducts(statment_form.SessionId)
	if err != nil {
		log.Printf("Неудалось правиьно найти RecProducts в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	
	}

	lead := Leads{
		LeadId: lead_id,
		SessionId: statment_form.SessionId,
		Name: statment_form.Contact.Name,
		Company: statment_form.Contact.Company,
		Email: statment_form.Contact.Email,
		Status: "new",
		RecProducts: rec_products,
		CreatedAt: created_at,
	}
	LogStruct("Созданная форма для записи в БД", lead)
	if err := globalDBCon.Create(&lead).Error; err != nil {
		log.Printf("Неудалось создать запись Заявки в БД | err: %s", err.Error())
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Insert DB Error", http.StatusInternalServerError, err))
		return
	}	

	var lead_form LeadForm
	lead_form.CreatedAt = created_at.GoString()
	lead_form.Status = "new"
	lead_form.LeadId = lead_id	
	LogStruct("Форма ответа для Front", lead)
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

	customer_id, err := GetCustomerIdFromDB(ctx)
	if err != nil {
		log.Printf("Ошибка при работе с Cookie | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}

	
	customer_record, err := GetOneFromDB[Customers]("customer_id = ?", customer_id, &Customers{})
	if err != nil {
		log.Printf("Ошибка при поиске пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.SetCookie(CookieName, "", -1, "/", "", false, true)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}

	if customer_record == nil {
		err = errors.New("Can't find Customer in DB")
		log.Printf("Неудалось найти пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}

	server_response := RegisterResponse{
		Login: customer_record.Login,
		IsAdmin: customer_record.IsAdmin,
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
		log.Printf("Ошибка при поиске пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}
	if customer_record == nil {
		err = errors.New("Can't find Customer in DB")
		log.Printf("Неудалось найти пользователя в БД | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
		return
	}
	err = bcrypt.CompareHashAndPassword([]byte(customer_record.Password), []byte(cust_data.Password)) 
	if err != nil {
		log.Printf("Неверный пароль | Путь : %s | Ошибка: %v", ctx.Request.URL.Path, err)
		err = errors.New("Invalid login or password")
		ctx.JSON(http.StatusInternalServerError, FillErrorForm("Search DB Error", http.StatusInternalServerError, err))
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