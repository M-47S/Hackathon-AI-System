package main


type ErrorForm struct {
	Type string `json:"type"`
	Title string `json:"title"`
	Status int `json:"status"`
	Detail string `json:"detail"`
}

// Структура для возвращения информации о сессии /api/sessions
type SessionForm struct{
	SessionId string `json:"session_id"`
	Status string `json:"status"`
	CreatedAt string `json:"created_at"`
}



// Структура для принятия запроса с Frontend /api/chat
type UserRequest struct{
	SessionId string `json:"session_id"`
	RequestId string `json:"request_id"`
	Message string `json:"message"`
}

// Структура для считывания []History из БД для LLM /v1/chat
type AiHistoryMessageForm struct {
	Role string `json:"role"`
	Content string `json:"content"`
}

// Структура для считывания []History из БД для Пользователя /v1/chat
type FrontHistoryMessageForm struct {
	Role string `json:"role"`
	Content string `json:"content"`
	Action string `json:"action"`
	RecProducts []string `json:"rec_products"`
	CreatedAt string `json:"created_at"`
}

type UserHistoryForm struct {
	SessionId string `json:"session_id"`
	History []FrontHistoryMessageForm `json:"history"`
}

// Структура для считывания customer_profile из БД для LLM /v1/chat
type CustomerProfileForm struct {
	BusinessType string `json:"business_type"`
	ObjectCount int `json:"object_count"`
	Problem string `json:"problem"`
	MeteringMethod string `json:"metering_method"`
}


// Структура для отправки запроса в LLM /v1/chat
type RequestToAI struct {
  SessionId string `json:"session_id"`
  RequestId string `json:"request_id"`
  Message  string `json:"message"`
  History []AiHistoryMessageForm `json:"history"`
  CustomerProfile CustomerProfileForm `json:"customer_profile"`
  ConversationSummary string `json:"conversation_summary"`
}
// Структура для принятия ответа из LLM /v1/chat
type AIResponseForm struct {
  SessionId string `json:"session_id"`
  Action string `json:"action"`
  Answer string `json:"answer"`
  RecProducts []string `json:"rec_products"`
  CustomerProfile CustomerProfileForm `json:"customer_profile"`
  ConversationSummary string `json:"conversation_summary"`
}

// Структура в которой хранится информация о пользователе которая призодит с Front /api/leads
type UserContactData struct {
	Name string `json"name"`
	Company string `json:"company"`
	Phone string `json:"phone"`
	Email string `json:"email"`
}

// Структура в которой хранится информация для создания Заявки /api/leads
type ManagerStatementForm struct {
	SessionId string `json:"session_id"`
	SubmissionId string `json:"submission_id"`
	Contact UserContactData `json:"contact"`
	Consent bool `json:"consent"`
}

// Структура которая возвращается с информацией о созданной заявке /api/leads
type LeadForm struct {
	LeadId string `json:"lead_id"`
	Status string `json:"status"`
	CreatedAt string `json:"created_at"`
}

type CustomerRegisterData struct {
	Login string `json:"login"`
	Password string `json:"password"`
	IsAdmin bool `json:"is_admin"`
}

type CustomerLoginData struct {
	Login string `json:"login"`
	Password string `json:"password"`
	IsAdmin bool `json:"is_admin"`
}

type RegisterResponse struct {
	Login string `json:"login"`
	IsAdmin bool `json:"is_admin"`
}