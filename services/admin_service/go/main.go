package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/mux"
)

const (
	DefaultPageSize = 20
	MaxPageSize     = 100
	CacheTTL        = 5 * time.Minute
)

var (
	environment    string
	jwtSecret      []byte
	dynamoClient   *dynamodb.Client
	tenantsTable   string
	projectsTable  string
	jobsTable      string
	auditLogTable  string
	// Simple in-memory cache (for production, use Redis/ElastiCache)
	cache          = make(map[string]cacheEntry)
)

type cacheEntry struct {
	data      interface{}
	expiresAt time.Time
}

type PaginationParams struct {
	Page     int
	PageSize int
}

type PaginatedResponse struct {
	Items      interface{} `json:"items"`
	Page       int         `json:"page"`
	PageSize   int         `json:"page_size"`
	Total      int         `json:"total"`
	TotalPages int         `json:"total_pages"`
}

type Tenant struct {
	TenantID   string    `json:"tenant_id" dynamodbav:"tenant_id"`
	Name       string    `json:"name" dynamodbav:"name"`
	CreatedAt  string    `json:"created_at" dynamodbav:"created_at"`
	UpdatedAt  string    `json:"updated_at" dynamodbav:"updated_at"`
}

type Project struct {
	ProjectID              string   `json:"project_id" dynamodbav:"project_id"`
	TenantID               string   `json:"tenant_id" dynamodbav:"tenant_id"`
	GitHubOrg              string   `json:"github_org" dynamodbav:"github_org"`
	GitHubRepo             string   `json:"github_repo" dynamodbav:"github_repo"`
	AllowedSubjectPatterns []string `json:"allowed_subject_patterns" dynamodbav:"allowed_subject_patterns"`
	CreatedAt              string   `json:"created_at" dynamodbav:"created_at"`
	UpdatedAt              string   `json:"updated_at" dynamodbav:"updated_at"`
}

type Job struct {
	JobID         string `json:"job_id" dynamodbav:"job_id"`
	TenantID      string `json:"tenant_id" dynamodbav:"tenant_id"`
	ProjectID     string `json:"project_id" dynamodbav:"project_id"`
	Repo          string `json:"repo" dynamodbav:"repo"`
	Status        string `json:"status" dynamodbav:"status"`
	CreatedAt     string `json:"created_at" dynamodbav:"created_at"`
	StartedAt     string `json:"started_at,omitempty" dynamodbav:"started_at,omitempty"`
	FinishedAt    string `json:"finished_at,omitempty" dynamodbav:"finished_at,omitempty"`
	FindingsCount int    `json:"findings_count" dynamodbav:"findings_count"`
	ReportCount   int    `json:"report_count,omitempty" dynamodbav:"report_count,omitempty"`
}

type AuditLog struct {
	LogID    string                 `json:"log_id" dynamodbav:"log_id"`
	TenantID string                 `json:"tenant_id" dynamodbav:"tenant_id"`
	Actor    string                 `json:"actor" dynamodbav:"actor"`
	Action   string                 `json:"action" dynamodbav:"action"`
	Result   string                 `json:"result" dynamodbav:"result"`
	Timestamp string                `json:"timestamp" dynamodbav:"timestamp"`
	Metadata map[string]interface{} `json:"metadata,omitempty" dynamodbav:"metadata,omitempty"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func init() {
	environment = getEnv("ENVIRONMENT", "dev")
	jwtSecretStr := getEnv("JWT_SECRET", "")
	if jwtSecretStr == "" {
		log.Fatal("JWT_SECRET environment variable is required")
	}
	jwtSecret = []byte(jwtSecretStr)

	tenantsTable = fmt.Sprintf("%s-nnast-tenants", environment)
	projectsTable = fmt.Sprintf("%s-nnast-projects", environment)
	jobsTable = fmt.Sprintf("%s-nnast-jobs", environment)
	auditLogTable = fmt.Sprintf("%s-nnast-audit-logs", environment)

	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatalf("Failed to load AWS config: %v", err)
	}
	dynamoClient = dynamodb.NewFromConfig(cfg)
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func verifyJWT(tokenString string) (*jwt.Token, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return jwtSecret, nil
	})
	return token, err
}

func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			respondError(w, "Authorization header required", http.StatusUnauthorized)
			return
		}

		parts := strings.Split(authHeader, " ")
		if len(parts) != 2 || parts[0] != "Bearer" {
			respondError(w, "Invalid authorization header format", http.StatusUnauthorized)
			return
		}

		token, err := verifyJWT(parts[1])
		if err != nil || !token.Valid {
			respondError(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		// Store token claims in context
		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			respondError(w, "Invalid token claims", http.StatusUnauthorized)
			return
		}

		ctx := context.WithValue(r.Context(), "tenant_id", claims["tenant_id"])
		ctx = context.WithValue(ctx, "project_id", claims["project_id"])
		next(w, r.WithContext(ctx))
	}
}

func respondError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(ErrorResponse{Error: message})
}

func parsePagination(r *http.Request) PaginationParams {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page < 1 {
		page = 1
	}

	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize < 1 {
		pageSize = DefaultPageSize
	}
	if pageSize > MaxPageSize {
		pageSize = MaxPageSize
	}

	return PaginationParams{Page: page, PageSize: pageSize}
}

// Tenant endpoints

func listTenantsHandler(w http.ResponseWriter, r *http.Request) {
	params := parsePagination(r)
	
	// Scan tenants table with pagination
	result, err := dynamoClient.Scan(context.TODO(), &dynamodb.ScanInput{
		TableName:     aws.String(tenantsTable),
		Limit:         aws.Int32(int32(params.PageSize)),
	})
	if err != nil {
		log.Printf("Failed to scan tenants: %v", err)
		respondError(w, "Failed to fetch tenants", http.StatusInternalServerError)
		return
	}

	var tenants []Tenant
	err = attributevalue.UnmarshalListOfMaps(result.Items, &tenants)
	if err != nil {
		log.Printf("Failed to unmarshal tenants: %v", err)
		respondError(w, "Failed to process tenants", http.StatusInternalServerError)
		return
	}

	// Calculate total (simplified - in production, use count or maintain counter)
	total := len(tenants)
	totalPages := (total + params.PageSize - 1) / params.PageSize

	response := PaginatedResponse{
		Items:      tenants,
		Page:       params.Page,
		PageSize:   params.PageSize,
		Total:      total,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func getTenantHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	tenantID := vars["tenant_id"]

	result, err := dynamoClient.GetItem(context.TODO(), &dynamodb.GetItemInput{
		TableName: aws.String(tenantsTable),
		Key: map[string]types.AttributeValue{
			"tenant_id": &types.AttributeValueMemberS{Value: tenantID},
		},
	})
	if err != nil {
		log.Printf("Failed to get tenant: %v", err)
		respondError(w, "Failed to fetch tenant", http.StatusInternalServerError)
		return
	}

	if len(result.Item) == 0 {
		respondError(w, "Tenant not found", http.StatusNotFound)
		return
	}

	var tenant Tenant
	err = attributevalue.UnmarshalMap(result.Item, &tenant)
	if err != nil {
		log.Printf("Failed to unmarshal tenant: %v", err)
		respondError(w, "Failed to process tenant", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tenant)
}

func createTenantHandler(w http.ResponseWriter, r *http.Request) {
	var tenant Tenant
	if err := json.NewDecoder(r.Body).Decode(&tenant); err != nil {
		respondError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if tenant.TenantID == "" || tenant.Name == "" {
		respondError(w, "tenant_id and name are required", http.StatusBadRequest)
		return
	}

	now := time.Now().UTC().Format(time.RFC3339)
	tenant.CreatedAt = now
	tenant.UpdatedAt = now

	item, err := attributevalue.MarshalMap(tenant)
	if err != nil {
		log.Printf("Failed to marshal tenant: %v", err)
		respondError(w, "Failed to create tenant", http.StatusInternalServerError)
		return
	}

	_, err = dynamoClient.PutItem(context.TODO(), &dynamodb.PutItemInput{
		TableName: aws.String(tenantsTable),
		Item:      item,
	})
	if err != nil {
		log.Printf("Failed to create tenant: %v", err)
		respondError(w, "Failed to create tenant", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(tenant)
}

// Project endpoints

func listProjectsHandler(w http.ResponseWriter, r *http.Request) {
	params := parsePagination(r)
	tenantID := r.URL.Query().Get("tenant_id")

	var result *dynamodb.QueryOutput
	var err error

	if tenantID != "" {
		// Use GSI for tenant-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(projectsTable),
			IndexName:              aws.String("tenant-id-index"),
			KeyConditionExpression: aws.String("tenant_id = :tenant_id"),
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":tenant_id": &types.AttributeValueMemberS{Value: tenantID},
			},
			Limit: aws.Int32(int32(params.PageSize)),
		})
	} else {
		// Scan all projects
		result, err = dynamoClient.Scan(context.TODO(), &dynamodb.ScanInput{
			TableName: aws.String(projectsTable),
			Limit:    aws.Int32(int32(params.PageSize)),
		})
	}

	if err != nil {
		log.Printf("Failed to query projects: %v", err)
		respondError(w, "Failed to fetch projects", http.StatusInternalServerError)
		return
	}

	var projects []Project
	err = attributevalue.UnmarshalListOfMaps(result.Items, &projects)
	if err != nil {
		log.Printf("Failed to unmarshal projects: %v", err)
		respondError(w, "Failed to process projects", http.StatusInternalServerError)
		return
	}

	total := len(projects)
	totalPages := (total + params.PageSize - 1) / params.PageSize

	response := PaginatedResponse{
		Items:      projects,
		Page:       params.Page,
		PageSize:   params.PageSize,
		Total:      total,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func getProjectHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	projectID := vars["project_id"]

	result, err := dynamoClient.GetItem(context.TODO(), &dynamodb.GetItemInput{
		TableName: aws.String(projectsTable),
		Key: map[string]types.AttributeValue{
			"project_id": &types.AttributeValueMemberS{Value: projectID},
		},
	})
	if err != nil {
		log.Printf("Failed to get project: %v", err)
		respondError(w, "Failed to fetch project", http.StatusInternalServerError)
		return
	}

	if len(result.Item) == 0 {
		respondError(w, "Project not found", http.StatusNotFound)
		return
	}

	var project Project
	err = attributevalue.UnmarshalMap(result.Item, &project)
	if err != nil {
		log.Printf("Failed to unmarshal project: %v", err)
		respondError(w, "Failed to process project", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(project)
}

func createProjectHandler(w http.ResponseWriter, r *http.Request) {
	var project Project
	if err := json.NewDecoder(r.Body).Decode(&project); err != nil {
		respondError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if project.ProjectID == "" || project.TenantID == "" || project.GitHubOrg == "" || project.GitHubRepo == "" {
		respondError(w, "project_id, tenant_id, github_org, and github_repo are required", http.StatusBadRequest)
		return
	}

	now := time.Now().UTC().Format(time.RFC3339)
	project.CreatedAt = now
	project.UpdatedAt = now

	item, err := attributevalue.MarshalMap(project)
	if err != nil {
		log.Printf("Failed to marshal project: %v", err)
		respondError(w, "Failed to create project", http.StatusInternalServerError)
		return
	}

	_, err = dynamoClient.PutItem(context.TODO(), &dynamodb.PutItemInput{
		TableName: aws.String(projectsTable),
		Item:      item,
	})
	if err != nil {
		log.Printf("Failed to create project: %v", err)
		respondError(w, "Failed to create project", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(project)
}

func updateProjectHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	projectID := vars["project_id"]

	var updates Project
	if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
		respondError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Get existing project
	result, err := dynamoClient.GetItem(context.TODO(), &dynamodb.GetItemInput{
		TableName: aws.String(projectsTable),
		Key: map[string]types.AttributeValue{
			"project_id": &types.AttributeValueMemberS{Value: projectID},
		},
	})
	if err != nil || len(result.Item) == 0 {
		respondError(w, "Project not found", http.StatusNotFound)
		return
	}

	var project Project
	attributevalue.UnmarshalMap(result.Item, &project)

	// Update fields
	if updates.GitHubOrg != "" {
		project.GitHubOrg = updates.GitHubOrg
	}
	if updates.GitHubRepo != "" {
		project.GitHubRepo = updates.GitHubRepo
	}
	if updates.AllowedSubjectPatterns != nil {
		project.AllowedSubjectPatterns = updates.AllowedSubjectPatterns
	}
	project.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	item, err := attributevalue.MarshalMap(project)
	if err != nil {
		log.Printf("Failed to marshal project: %v", err)
		respondError(w, "Failed to update project", http.StatusInternalServerError)
		return
	}

	_, err = dynamoClient.PutItem(context.TODO(), &dynamodb.PutItemInput{
		TableName: aws.String(projectsTable),
		Item:      item,
	})
	if err != nil {
		log.Printf("Failed to update project: %v", err)
		respondError(w, "Failed to update project", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(project)
}

func deleteProjectHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	projectID := vars["project_id"]

	_, err := dynamoClient.DeleteItem(context.TODO(), &dynamodb.DeleteItemInput{
		TableName: aws.String(projectsTable),
		Key: map[string]types.AttributeValue{
			"project_id": &types.AttributeValueMemberS{Value: projectID},
		},
	})
	if err != nil {
		log.Printf("Failed to delete project: %v", err)
		respondError(w, "Failed to delete project", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// Job endpoints

func listJobsHandler(w http.ResponseWriter, r *http.Request) {
	params := parsePagination(r)
	tenantID := r.URL.Query().Get("tenant_id")
	projectID := r.URL.Query().Get("project_id")
	status := r.URL.Query().Get("status")

	var result *dynamodb.QueryOutput
	var err error

	if tenantID != "" {
		// Use GSI for tenant-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(jobsTable),
			IndexName:              aws.String("tenant-id-index"),
			KeyConditionExpression: aws.String("tenant_id = :tenant_id"),
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":tenant_id": &types.AttributeValueMemberS{Value: tenantID},
			},
			ScanIndexForward: aws.Bool(false), // Descending order
			Limit:            aws.Int32(int32(params.PageSize)),
		})
	} else if projectID != "" {
		// Use GSI for project-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(jobsTable),
			IndexName:              aws.String("project-id-index"),
			KeyConditionExpression: aws.String("project_id = :project_id"),
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":project_id": &types.AttributeValueMemberS{Value: projectID},
			},
			ScanIndexForward: aws.Bool(false),
			Limit:            aws.Int32(int32(params.PageSize)),
		})
	} else if status != "" {
		// Use GSI for status-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(jobsTable),
			IndexName:              aws.String("status-created-index"),
			KeyConditionExpression: aws.String("#status = :status"),
			ExpressionAttributeNames: map[string]string{
				"#status": "status",
			},
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":status": &types.AttributeValueMemberS{Value: status},
			},
			ScanIndexForward: aws.Bool(false),
			Limit:            aws.Int32(int32(params.PageSize)),
		})
	} else {
		// Scan all jobs
		result, err = dynamoClient.Scan(context.TODO(), &dynamodb.ScanInput{
			TableName: aws.String(jobsTable),
			Limit:    aws.Int32(int32(params.PageSize)),
		})
	}

	if err != nil {
		log.Printf("Failed to query jobs: %v", err)
		respondError(w, "Failed to fetch jobs", http.StatusInternalServerError)
		return
	}

	var jobs []Job
	err = attributevalue.UnmarshalListOfMaps(result.Items, &jobs)
	if err != nil {
		log.Printf("Failed to unmarshal jobs: %v", err)
		respondError(w, "Failed to process jobs", http.StatusInternalServerError)
		return
	}

	total := len(jobs)
	totalPages := (total + params.PageSize - 1) / params.PageSize

	response := PaginatedResponse{
		Items:      jobs,
		Page:       params.Page,
		PageSize:   params.PageSize,
		Total:      total,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func getJobHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["job_id"]

	result, err := dynamoClient.GetItem(context.TODO(), &dynamodb.GetItemInput{
		TableName: aws.String(jobsTable),
		Key: map[string]types.AttributeValue{
			"job_id": &types.AttributeValueMemberS{Value: jobID},
		},
	})
	if err != nil {
		log.Printf("Failed to get job: %v", err)
		respondError(w, "Failed to fetch job", http.StatusInternalServerError)
		return
	}

	if len(result.Item) == 0 {
		respondError(w, "Job not found", http.StatusNotFound)
		return
	}

	var job Job
	err = attributevalue.UnmarshalMap(result.Item, &job)
	if err != nil {
		log.Printf("Failed to unmarshal job: %v", err)
		respondError(w, "Failed to process job", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(job)
}

// Audit log endpoints

func listAuditLogsHandler(w http.ResponseWriter, r *http.Request) {
	params := parsePagination(r)
	tenantID := r.URL.Query().Get("tenant_id")
	actor := r.URL.Query().Get("actor")

	var result *dynamodb.QueryOutput
	var err error

	if tenantID != "" {
		// Use GSI for tenant-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(auditLogTable),
			IndexName:              aws.String("tenant-timestamp-index"),
			KeyConditionExpression: aws.String("tenant_id = :tenant_id"),
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":tenant_id": &types.AttributeValueMemberS{Value: tenantID},
			},
			ScanIndexForward: aws.Bool(false), // Descending order (newest first)
			Limit:            aws.Int32(int32(params.PageSize)),
		})
	} else if actor != "" {
		// Use GSI for actor-based query
		result, err = dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
			TableName:              aws.String(auditLogTable),
			IndexName:              aws.String("actor-timestamp-index"),
			KeyConditionExpression: aws.String("actor = :actor"),
			ExpressionAttributeValues: map[string]types.AttributeValue{
				":actor": &types.AttributeValueMemberS{Value: actor},
			},
			ScanIndexForward: aws.Bool(false),
			Limit:            aws.Int32(int32(params.PageSize)),
		})
	} else {
		// Scan all audit logs
		result, err = dynamoClient.Scan(context.TODO(), &dynamodb.ScanInput{
			TableName: aws.String(auditLogTable),
			Limit:    aws.Int32(int32(params.PageSize)),
		})
	}

	if err != nil {
		log.Printf("Failed to query audit logs: %v", err)
		respondError(w, "Failed to fetch audit logs", http.StatusInternalServerError)
		return
	}

	var logs []AuditLog
	err = attributevalue.UnmarshalListOfMaps(result.Items, &logs)
	if err != nil {
		log.Printf("Failed to unmarshal audit logs: %v", err)
		respondError(w, "Failed to process audit logs", http.StatusInternalServerError)
		return
	}

	total := len(logs)
	totalPages := (total + params.PageSize - 1) / params.PageSize

	response := PaginatedResponse{
		Items:      logs,
		Page:       params.Page,
		PageSize:   params.PageSize,
		Total:      total,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func main() {
	r := mux.NewRouter()

	// Health check (no auth required)
	r.HandleFunc("/health", healthHandler).Methods("GET")

	// API routes (auth required)
	api := r.PathPrefix("/api/v1").Subrouter()
	api.Use(authMiddleware)

	// Tenant routes
	api.HandleFunc("/tenants", listTenantsHandler).Methods("GET")
	api.HandleFunc("/tenants", createTenantHandler).Methods("POST")
	api.HandleFunc("/tenants/{tenant_id}", getTenantHandler).Methods("GET")

	// Project routes
	api.HandleFunc("/projects", listProjectsHandler).Methods("GET")
	api.HandleFunc("/projects", createProjectHandler).Methods("POST")
	api.HandleFunc("/projects/{project_id}", getProjectHandler).Methods("GET")
	api.HandleFunc("/projects/{project_id}", updateProjectHandler).Methods("PUT")
	api.HandleFunc("/projects/{project_id}", deleteProjectHandler).Methods("DELETE")

	// Job routes
	api.HandleFunc("/jobs", listJobsHandler).Methods("GET")
	api.HandleFunc("/jobs/{job_id}", getJobHandler).Methods("GET")

	// Audit log routes
	api.HandleFunc("/audit-logs", listAuditLogsHandler).Methods("GET")

	// CORS middleware
	corsMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			next.ServeHTTP(w, r)
		})
	}

	port := getEnv("PORT", "8080")
	log.Printf("Starting admin service on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, corsMiddleware(r)))
}
