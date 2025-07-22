import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


np.random.seed(42)


def generate_salary_dataset(size=5000):
    
    tech_jobs = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'UX Designer']
    finance_jobs = ['Financial Analyst', 'Accountant', 'Investment Banker', 'Risk Manager']
    healthcare_jobs = ['Doctor', 'Nurse', 'Pharmacist', 'Medical Technician']
    other_jobs = ['Teacher', 'Marketing Manager', 'HR Specialist', 'Sales Executive']
    
    all_jobs = tech_jobs + finance_jobs + healthcare_jobs + other_jobs
    
   
    educations = ['High School', 'Bachelor', 'Master', 'PhD']
    
    
    locations = {
        'New York': 1.8,
        'San Francisco': 1.9,
        'Chicago': 1.2,
        'Austin': 1.1,
        'Seattle': 1.4,
        'Boston': 1.5,
        'Los Angeles': 1.7,
        'Denver': 1.3,
        'Atlanta': 1.0,
        'Remote': 1.0
    }
    
  
    company_sizes = ['Startup (1-50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000+)']
    
   
    tech_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript']
    business_skills = ['Excel', 'PowerPoint', 'Financial Modeling', 'Project Management']
    soft_skills = ['Leadership', 'Communication', 'Teamwork']
    all_skills = tech_skills + business_skills + soft_skills
    
  
    data = []
    for _ in range(size):
       
        age = np.random.randint(22, 60)
        experience = np.random.randint(0, 40)
        education = np.random.choice(educations, p=[0.1, 0.5, 0.3, 0.1])
        job = np.random.choice(all_jobs)
        location = np.random.choice(list(locations.keys()))
        company_size = np.random.choice(company_sizes)
        
        
        if job in tech_jobs:
            base = 60000 + experience * 2500 + np.random.randint(-5000, 10000)
        elif job in finance_jobs:
            base = 55000 + experience * 2000 + np.random.randint(-5000, 8000)
        elif job in healthcare_jobs:
            base = 50000 + experience * 3000 + np.random.randint(-5000, 15000)
        else:
            base = 45000 + experience * 1800 + np.random.randint(-5000, 5000)
        
     
        if education == 'Bachelor':
            base *= 1.0
        elif education == 'Master':
            base *= 1.2
        elif education == 'PhD':
            base *= 1.4
        else:
            base *= 0.8
       
        base *= locations[location]
        
      
        if company_size == 'Startup (1-50)':
            base *= 0.9
        elif company_size == 'Small (50-200)':
            base *= 1.0
        elif company_size == 'Medium (200-1000)':
            base *= 1.1
        else:
            base *= 1.3
            
   
        num_skills = np.random.randint(5, 9)
        skills = np.random.choice(all_skills, num_skills, replace=False)
       
        skill_bonus = 0
        for skill in skills:
            if skill in ['Machine Learning', 'AWS', 'Financial Modeling']:
                skill_bonus += 3000
            elif skill in ['Python', 'SQL', 'Project Management']:
                skill_bonus += 2500
            elif skill in ['Leadership', 'JavaScript']:
                skill_bonus += 1500
            else:
                skill_bonus += 500
                
        base += skill_bonus
        
        salary = int(base + np.random.randint(-5000, 5000))
        
        row = {
            'age': age,
            'experience': experience,
            'education': education,
            'job_title': job,
            'location': location,
            'company_size': company_size,
            'skills': ', '.join(skills),
            'salary': salary
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

df = generate_salary_dataset(10000)
df.to_csv('salary_data.csv', index=False)