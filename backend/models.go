package main

type SessionForm struct{
	session_id string `json:"session_id"`
	status string `json:"status"`
	created_at string `json:created_at`
}


// Структура для /api/chat с сообщением пользователя
type UserRequest struct{
	session_id string `json:"session_id"`
	request_id string `json:request_id""`
	message string `json:"message"`
}

type HistoryForm struct {
	role string `json:"role"`
	content string `json:"content"`
}

type CustomerProfileForm struct {
	business_type string `json:"business_type"`
	object_count int `json:"object_count"`
	problem string `json:"problem"`
	metering_method string `json:"metering_method"`
}

type RequestToAI struct {
  session_id string `json:"session_id"`
  request_id string `json:"request_id"`
  message  string `json:"message"`
  history []HistoryForm `json:"history"`
  customer_profile CustomerProfileForm `json:"customer_profile"`
  conversation_summary string `json:"conversation_summary"`
}

type AIResponseForm struct {
  action string `json:"action"`
  answer string `json:"answer"`
  product_ids []string `json:"products_ids"`
  customer_profile CustomerProfileForm `json:"customer_profile"`
  conversation_summary string `json:"conversation_summary"`
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