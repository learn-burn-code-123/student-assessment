/**
 * Main JavaScript for Student Assessment System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form validation and submission
    const assessmentForm = document.getElementById('assessmentForm');
    if (assessmentForm) {
        initializeAssessmentForm();
    }

    // Report page functionality
    const reportPage = document.getElementById('summary');
    if (reportPage) {
        initializeReportPage();
    }

    // Initialize navigation
    initializeNavigation();
});

/**
 * Initialize the assessment form
 */
function initializeAssessmentForm() {
    const form = document.getElementById('assessmentForm');
    const progressBar = document.getElementById('progressBar');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const inputs = form.querySelectorAll('input, select');
    const totalInputs = inputs.length;
    
    // Update progress bar as user fills out the form
    inputs.forEach(input => {
        input.addEventListener('change', updateProgress);
    });
    
    function updateProgress() {
        let filledInputs = 0;
        inputs.forEach(input => {
            if (input.value.trim() !== '') {
                filledInputs++;
            }
        });
        
        const progress = Math.round((filledInputs / totalInputs) * 100);
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        if (!validateForm()) {
            return;
        }
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Collect form data
        const formData = {};
        inputs.forEach(input => {
            formData[input.name] = input.value;
        });
        
        // Send data to server
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.redirect) {
                window.location.href = data.redirect;
            } else {
                alert('提交失败，请重试');
                loadingOverlay.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('发生错误，请重试');
            loadingOverlay.style.display = 'none';
        });
    });
    
    // Form validation
    function validateForm() {
        let isValid = true;
        
        inputs.forEach(input => {
            if (input.hasAttribute('required') && input.value.trim() === '') {
                isValid = false;
                input.classList.add('is-invalid');
                
                // Create error message if it doesn't exist
                let errorDiv = input.nextElementSibling;
                if (!errorDiv || !errorDiv.classList.contains('invalid-feedback')) {
                    errorDiv = document.createElement('div');
                    errorDiv.className = 'invalid-feedback';
                    errorDiv.textContent = '此字段为必填项';
                    input.parentNode.insertBefore(errorDiv, input.nextSibling);
                }
            } else {
                input.classList.remove('is-invalid');
                
                // Remove error message if it exists
                let errorDiv = input.nextElementSibling;
                if (errorDiv && errorDiv.classList.contains('invalid-feedback')) {
                    errorDiv.remove();
                }
            }
        });
        
        if (!isValid) {
            // Scroll to first invalid input
            const firstInvalid = form.querySelector('.is-invalid');
            if (firstInvalid) {
                firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
        
        return isValid;
    }
}

/**
 * Initialize the report page
 */
function initializeReportPage() {
    // Set report date
    const reportDateElement = document.getElementById('reportDate');
    if (reportDateElement) {
        const now = new Date();
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        reportDateElement.textContent = now.toLocaleDateString('zh-CN', options);
    }
    
    // Print button functionality
    const printBtn = document.querySelector('.print-btn');
    if (printBtn) {
        printBtn.addEventListener('click', function() {
            window.print();
        });
    }
}

/**
 * Initialize navigation
 */
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Smooth scroll to sections
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Only handle internal links
            if (href.startsWith('#')) {
                e.preventDefault();
                
                // Update active link
                navLinks.forEach(l => l.classList.remove('active'));
                this.classList.add('active');
                
                // Scroll to section
                const targetSection = document.querySelector(href);
                if (targetSection) {
                    window.scrollTo({
                        top: targetSection.offsetTop - 100,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
    
    // Update active nav link on scroll
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('.section-card, .report-section');
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 150;
            const sectionHeight = section.offsetHeight;
            
            if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
                currentSection = '#' + section.getAttribute('id');
            }
        });
        
        if (currentSection !== '') {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === currentSection) {
                    link.classList.add('active');
                }
            });
        }
    });
}
