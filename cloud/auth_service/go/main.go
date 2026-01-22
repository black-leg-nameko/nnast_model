package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/golang-jwt/jwt/v5"
	"github.com/lestrrat-go/jwx/v2/jwk"
	jwtlib "github.com/lestrrat-go/jwx/v2/jwt"
)

const (
	GitHubOIDCIssuer  = "https://token.actions.githubusercontent.com"
	GitHubOIDCAudience = "nnast-cloud"
	JWTExpirationMinutes = 15
)

var (
	environment   string
	jwtSecret     []byte
	dynamoClient  *dynamodb.Client
	auditLogTable string
	projectsTable string
)

type AuthRequest struct {
	GitHubOIDCToken string `json:"github_oidc_token"`
}

type AuthResponse struct {
	ShortLivedJWT string `json:"short_lived_jwt"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

type Project struct {
	ProjectID              string   `dynamodbav:"project_id"`
	TenantID               string   `dynamodbav:"tenant_id"`
	GitHubOrg              string   `dynamodbav:"github_org"`
	GitHubRepo             string   `dynamodbav:"github_repo"`
	AllowedSubjectPatterns []string `dynamodbav:"allowed_subject_patterns"`
}

func init() {
	environment = getEnv("ENVIRONMENT", "dev")
	jwtSecretStr := getEnv("JWT_SECRET", "")
	if jwtSecretStr == "" {
		log.Fatal("JWT_SECRET environment variable is required")
	}
	jwtSecret = []byte(jwtSecretStr)

	auditLogTable = fmt.Sprintf("%s-nnast-audit-logs", environment)
	projectsTable = fmt.Sprintf("%s-nnast-projects", environment)

	// Configure AWS SDK
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

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func verifyGitHubOIDCToken(tokenString string) (jwtlib.Token, error) {
	// Get GitHub OIDC JWKS
	jwksURL := fmt.Sprintf("%s/.well-known/jwks.json", GitHubOIDCIssuer)
	keySet, err := jwk.Fetch(context.Background(), jwksURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch JWKS: %w", err)
	}

	// Verify token
	token, err := jwtlib.Parse(
		[]byte(tokenString),
		jwtlib.WithKeySet(keySet),
		jwtlib.WithValidate(true),
		jwtlib.WithIssuer(GitHubOIDCIssuer),
		jwtlib.WithAudience(GitHubOIDCAudience),
	)
	if err != nil {
		return nil, fmt.Errorf("token verification failed: %w", err)
	}

	return token, nil
}

func getProjectByRepo(githubOrg, githubRepo string) (*Project, error) {
	// Get project from DynamoDB
	result, err := dynamoClient.Query(context.TODO(), &dynamodb.QueryInput{
		TableName:              aws.String(projectsTable),
		IndexName:              aws.String("repo-index"),
		KeyConditionExpression: aws.String("github_org = :org AND github_repo = :repo"),
		ExpressionAttributeValues: map[string]types.AttributeValue{
			":org":  &types.AttributeValueMemberS{Value: githubOrg},
			":repo": &types.AttributeValueMemberS{Value: githubRepo},
		},
	})
	if err != nil {
		return nil, err
	}

	if len(result.Items) == 0 {
		return nil, nil
	}

	var project Project
	err = attributevalue.UnmarshalMap(result.Items[0], &project)
	if err != nil {
		return nil, err
	}

	return &project, nil
}

func checkSubjectAllowed(subject string, project *Project) bool {
	allowedPatterns := project.AllowedSubjectPatterns
	if len(allowedPatterns) == 0 {
		// Default: allow repo:{org}/{repo}:*
		return true
	}

	for _, pattern := range allowedPatterns {
		// Simple wildcard matching
		patternRegex := strings.ReplaceAll(pattern, "*", ".*")
		matched, _ := regexp.MatchString(patternRegex, subject)
		if matched {
			return true
		}
	}

	return false
}

func generateJWT(tenantID, projectID, repo, subject string) (string, error) {
	now := time.Now()
	exp := now.Add(JWTExpirationMinutes * time.Minute)

	claims := jwt.MapClaims{
		"iss":       "nnast-cloud",
		"sub":       subject,
		"aud":       "nnast-api",
		"exp":       exp.Unix(),
		"iat":       now.Unix(),
		"tenant_id": tenantID,
		"project_id": projectID,
		"repo":      repo,
		"scopes":    []string{"report:generate"},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(jwtSecret)
}

func logAuditEvent(tenantID, actor, action, result string, metadata map[string]interface{}) {
	logID := fmt.Sprintf("%d-%s", time.Now().UnixMilli(), sha256Hash(actor)[:8])
	timestamp := time.Now().UTC().Format(time.RFC3339)

	item := map[string]types.AttributeValue{
		"log_id":    &types.AttributeValueMemberS{Value: logID},
		"tenant_id": &types.AttributeValueMemberS{Value: tenantID},
		"actor":     &types.AttributeValueMemberS{Value: actor},
		"action":    &types.AttributeValueMemberS{Value: action},
		"result":    &types.AttributeValueMemberS{Value: result},
		"timestamp": &types.AttributeValueMemberS{Value: timestamp},
		"ttl":       &types.AttributeValueMemberN{Value: fmt.Sprintf("%d", time.Now().Add(365*24*time.Hour).Unix())},
	}

	if metadata != nil {
		metadataJSON, _ := json.Marshal(metadata)
		item["metadata"] = &types.AttributeValueMemberS{Value: string(metadataJSON)}
	}

	_, err := dynamoClient.PutItem(context.TODO(), &dynamodb.PutItemInput{
		TableName: aws.String(auditLogTable),
		Item:      item,
	})
	if err != nil {
		log.Printf("Failed to log audit event: %v", err)
	}
}

func sha256Hash(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h)
}

func authOIDCHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AuthRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		respondError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.GitHubOIDCToken == "" {
		respondError(w, "github_oidc_token is required", http.StatusBadRequest)
		return
	}

	// Verify OIDC token
	token, err := verifyGitHubOIDCToken(req.GitHubOIDCToken)
	if err != nil {
		log.Printf("OIDC token verification failed: %v", err)
		respondError(w, "Invalid OIDC token", http.StatusUnauthorized)
		return
	}

	// Get subject
	subject, ok := token.Get("sub")
	if !ok {
		respondError(w, "Invalid token subject", http.StatusUnauthorized)
		return
	}

	subjectStr := subject.(string)
	if subjectStr == "" {
		respondError(w, "Invalid token subject", http.StatusUnauthorized)
		return
	}

	// Extract repository information from subject
	// Format: repo:{org}/{repo}:ref:refs/heads/{branch}
	parts := strings.Split(subjectStr, ":")
	if len(parts) < 2 || parts[0] != "repo" {
		respondError(w, "Invalid subject format", http.StatusUnauthorized)
		return
	}

	repoPath := parts[1] // {org}/{repo}
	repoParts := strings.Split(repoPath, "/")
	if len(repoParts) != 2 {
		respondError(w, "Invalid repo format", http.StatusUnauthorized)
		return
	}

	githubOrg := repoParts[0]
	githubRepo := repoParts[1]

	// Get project
	project, err := getProjectByRepo(githubOrg, githubRepo)
	if err != nil {
		log.Printf("Failed to get project: %v", err)
		respondError(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if project == nil {
		respondError(w, "Project not found", http.StatusNotFound)
		return
	}

	// Check if subject is allowed
	if !checkSubjectAllowed(subjectStr, project) {
		logAuditEvent(
			project.TenantID,
			subjectStr,
			"auth",
			"denied",
			map[string]interface{}{"reason": "subject_not_allowed"},
		)
		respondError(w, "Subject not allowed", http.StatusForbidden)
		return
	}

	// Generate JWT
	shortJWT, err := generateJWT(project.TenantID, project.ProjectID, repoPath, subjectStr)
	if err != nil {
		log.Printf("Failed to generate JWT: %v", err)
		respondError(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Audit log
	logAuditEvent(
		project.TenantID,
		subjectStr,
		"auth",
		"success",
		map[string]interface{}{
			"project_id": project.ProjectID,
			"repo":       repoPath,
		},
	)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AuthResponse{ShortLivedJWT: shortJWT})
}

func respondError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(ErrorResponse{Error: message})
}

func main() {
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/auth/oidc", authOIDCHandler)

	port := getEnv("PORT", "8080")
	log.Printf("Starting auth service on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
