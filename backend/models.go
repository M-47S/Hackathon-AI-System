package main




var ProductMap = map[string]ProductData{
	"cloud_askue": {
		Id:          "cloud_askue",
		Name:        "Облачная АСКУЭ",
		Description: "Автоматизированный сбор и обработка показаний приборов учёта электроэнергии. Работает с 300+ типами счётчиков, позволяет видеть расход по объектам, арендаторам и цехам в одном кабинете.",
	},
	"solar_station": {
		Id:          "solar_station",
		Name:        "Солнечные электростанции",
		Description: "Сетевые (снижение платежа) и гибридные (с накопителями, для автономии) СЭС мощностью 15–500 кВт.",
	},
	"orem": {
		Id:          "orem",
		Name:        "Выход на оптовый рынок электроэнергии (ОРЕМ)",
		Description: "Переход на оптовый рынок для предприятий с присоединённой мощностью от 750 кВт. Включает аудит существующего учёта.",
	},
	"loss_detection": {
		Id:          "loss_detection",
		Name:        "Поиск коммерческих потерь",
		Description: "ML-сервис приоритизации выездных проверок для электросетевых компаний. Помогает находить хищения и вмешательства в счётчики.",
	},
	"photo_meter_reading": {
		Id:          "photo_meter_reading",
		Name:        "Распознавание показаний по фотографии",
		Description: "Для приборов, не подключённых к автоматике. Электроэнергия и вода. Распознаёт показания, серийный номер, штрих-код, год выпуска.",
	},
	"audit": {
		Id:          "audit",
		Name:        "Аудит учёта и расчёт экономического эффекта",
		Description: "Проверка существующей системы учёта и оценка потенциальной экономии до внедрения решения.",
	},
}


type ProductData struct {
	Id string `json:"id"`
	Name string `json:"name"`
	Description string `json:"description"`
}


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

type MessageForm struct {
	Id string `json:"id"`
	Role string `json:"role"`
	Content string `json:"content"`
	CreatedAt string `json:"created_at"`
}

// ответ на фронт от /api/chat
type FromAiMessageResponse struct {
	SessionId string `json:"session_id"`
	RequestId string `json:"request_id"`
	Message MessageForm `json:"message"`
	Action string `json:"action"`
	RecProducts []ProductData `json:"products"`
}

// Структура для считывания []History из БД для Пользователя /v1/chat
type FrontHistoryMessageForm struct {
	Role string `json:"role"`
	Content string `json:"content"`
	Action string `json:"action"`
	RecProducts []string `json:"recommended_product_ids"`
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
  RecProducts []string `json:"recommended_product_ids"`
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

type SessionInfo struct {
	SessionId string `json:"session_id"`
	ConversationSummary string `json:"conversation_summary"`
}

type RegisterResponse struct {
	Login string `json:"login"`
	IsAdmin bool `json:"is_admin"`
	Sessions []SessionInfo `json:"sessions"`
}