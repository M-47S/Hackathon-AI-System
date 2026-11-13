package main

import "time"

// Тут описаны таблицы а также связи между ними
type Customers struct {
	CustomerId string `gorm:"primaryKey;type:uuid;default:gen_random_uuid()" json:"customer_id"`
	Login string `gorm:"type:text;not null;uniqueIndex" json:"login"`
	Password string `gorm:"type:text;not null" json:"password"`
	IsAdmin bool `gorm:"type:bool;not null" json:"password"`
	Cookies []Cookies `gorm:"foreignKey:CustomerId;constraint:OnDelete:CASCADE;" json:"cookies"`
	Sessions []Session `gorm:"foreignKey:CustomerId;constraint:OnDelete:CASCADE;" json:"sessions"`
}

type Cookies struct {
	CookieId string `gorm:"primaryKey;type:uuid" json:"cookie_id"`
	CustomerId string `gorm:"type:uuid;not null" json:"customer_id"`
	ExpiresAt time.Time `gorm:"type:timestamptz;not null" json:"created_at"`
}

type Session struct{
	SessionId string `gorm:"type:uuid;primaryKey" json:"session_id"`
	CustomerId string `gorm:"type:uuid" json:"customer_id"`
	CreatedAt time.Time `gorm:"type:timestamptz;not null" json:"created_at"`
	
	Messages []Messages `gorm:"foreignKey:SessionId;constraint:OnDelete:CASCADE;" json:"messages"`
	Profile Profiles `gorm:"foreignKey:SessionId;constraint:OnDelete:CASCADE;" json:"profile"`
	Leads []Leads `gorm:"foreignKey:SessionId;constraint:OnDelete:CASCADE;" json:"leads"`
}

type Messages struct {
	MessageId string `gorm:"primaryKey;type:uuid" json:"message_id"` 
	SessionId string `gorm:"type:uuid;not null;index" json:"session_id"`
	Role string `gorm:"type:varchar(20);not null" json:"role"`
	Action string `gorm:"type:text" json:"action"`
	Content string `gorm:"type:text;not null" json:"content"`
	CreatedAt time.Time `gorm:"type:timestamptz;not null" json:"created_at"`
}

type Profiles struct {
	SessionId string `gorm:"type:uuid;primaryKey" json:"session_id"`
	ObjectCount int `gorm:"type:int;not null" json:"object_count"`
	Problem string `gorm:"type:text;not null" json:"problem"`
	BusinessType string `gorm:"type:text;not null" json:"business_type"`
	MeteringMethod string `gorm:"type:text" json:"metering_method"`
}

type Leads struct {
	LeadId string `gorm:"primaryKey;type:uuid" json:"lead_id"`
	SessionId string `gorm:"type:uuid;not null" json:"session_id"`
	Name string `gorm:"type:text" json:"name"`
	Company string `gorm:"type:text" json:"company"`
	Email string `gorm:"type:text" json:"email"`
	Phone string `gorm:"type:text" json:"phone"`
	Status string `gorm:"type:text" json:"status"`
	RecProducts []string `gorm:"type:text[]" json:"recommended_product_ids"`
	CreatedAt time.Time `gorm:"type:timestamptz;not null" json:"created_at"`
}