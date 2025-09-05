<div align="center">

# 🏠✨ **Stable Diffusion House Modifier**

*Transform Your Architectural Dreams into Reality with AI*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-3.5-purple.svg?style=for-the-badge)](https://stability.ai)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

<br>

**🚀 [Live Demo]((https://build-a-box-house-ai.onrender.com/))**

---

*A revolutionary AI-powered platform that combines the magic of Stable Diffusion 3.5 with Google Gemini's intelligence to transform, enhance, and reimagine house imagery like never before.*

</div>

---

## 🌟 **Why Choose Our House Modifier?**

<table>
<tr>
<td width="50%">

### 🎯 **Intelligent & Intuitive**
- **Smart AI Analysis** with Google Gemini
- **Context-Aware Processing** for optimal results
- **Natural Language Understanding** - just describe what you want
- **Automatic Method Selection** based on your intent

</td>
<td width="50%">

### ⚡ **Powerful & Professional**
- **State-of-the-art** Stable Diffusion 3.5 integration
- **Multiple Processing Methods** for every use case
- **Batch Processing** for efficiency
- **High-Quality Output** with customizable parameters

</td>
</tr>
</table>

---

## 🎨 **Spectacular Features**

<div align="center">
<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/100/000000/image-gallery.png" alt="Text-to-Image">
<h3>🖼️ Text-to-Image</h3>
<p><em>Generate stunning house images from simple descriptions</em></p>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/100/000000/magic-wand.png" alt="Background Replacement">
<h3>🌅 Background Magic</h3>
<p><em>Replace environments while preserving house structure</em></p>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/100/000000/paint-palette.png" alt="Inpainting">
<h3>🎨 Smart Inpainting</h3>
<p><em>Remove or modify specific areas with precision</em></p>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/100/000000/resize.png" alt="Outpainting">
<h3>📐 Extend Boundaries</h3>
<p><em>Expand images beyond original limits</em></p>
</td>
</tr>
</table>
</div>

### 🔥 **Advanced Capabilities**

> **🎭 Style Transfer** • Transform houses with artistic styles  
> **🏗️ Architectural Focus** • Specialized for building imagery  
> **🔄 Image-to-Image** • Reimagine existing photos  
> **🎯 Seed Control** • Reproducible and consistent results  

---

## 🚀 **Quick Start Journey**

### 📋 Prerequisites

<div align="center">

| Requirement | Version | Get It |
|-------------|---------|---------|
| ![Python](https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white) | 3.8+ | [Download Python](https://python.org) |
| ![API](https://img.shields.io/badge/Stability%20AI-API-purple?style=flat) | Latest | [Get API Key](https://platform.stability.ai/) |
| ![Gemini](https://img.shields.io/badge/Google%20Gemini-API-orange?style=flat&logo=google) | Latest | [Get API Key](https://makersuite.google.com/app/apikey) |

</div>

### ⚡ **Installation in 3 Steps**

```bash
# 🔽 Step 1: Clone the magic
git clone https://github.com/yourusername/stable-diffusion-house-modifier.git
cd stable-diffusion-house-modifier

# 📦 Step 2: Install dependencies
pip install -r requirements.txt

# 🚀 Step 3: Launch the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 🔑 **API Configuration**

```python
# 🔧 In main.py - Add your API keys
STABILITY_API_KEY = "your_stability_api_key_here"  # 🔑 Stability AI
GEMINI_API_KEY = "your_gemini_api_key_here"        # 🤖 Google Gemini
```

<div align="center">
<strong>🌐 Open <code>http://localhost:8000</code> and start creating!</strong>
</div>

---

## 🛠️ **API Arsenal**

<details>
<summary><strong>🔥 Core Processing Endpoints</strong></summary>

| Endpoint | Method | Description | Magic Level |
|----------|---------|------------|-------------|
| `/chat` | POST | 🤖 Main AI processing hub | ⭐⭐⭐⭐⭐ |
| `/upload` | POST | 📤 House image upload | ⭐⭐⭐⭐ |
| `/outpaint` | POST | 📐 Boundary extension | ⭐⭐⭐⭐⭐ |
| `/inpaint` | POST | 🎨 Selective modification | ⭐⭐⭐⭐⭐ |
| `/style-transfer` | POST | 🎭 Artistic transformation | ⭐⭐⭐⭐ |
| `/text-to-image` | POST | 🖼️ Text to visual magic | ⭐⭐⭐⭐⭐ |

</details>

<details>
<summary><strong>🔧 Utility & Management</strong></summary>

| Endpoint | Method | Description |
|----------|---------|------------|
| `/list-house-images` | GET | 📋 Available images |
| `/health` | GET | ❤️ System status |
| `/download/{session_id}` | GET | ⬇️ Result download |

</details>

---

## 💻 **Beautiful Web Interface**

<div align="center">

### 🎯 **Multiple Specialized Pages**

| Page | Purpose | Features |
|------|---------|----------|
| **🏠 Main Dashboard** | Central command center | All-in-one operations |
| **🎨 Style Transfer** | Artistic transformations | Multiple art styles |
| **✂️ Mask Editor** | Precision editing | Advanced masking tools |
| **📝 Text-to-Image** | Creative generation | Natural language input |

</div>

---

## ⚙️ **Configuration Paradise**

<table>
<tr>
<td width="50%">

### 🌍 **Environment Setup**
```bash
# 🔐 Secure API key management
STABILITY_API_KEY=your_stability_key
GEMINI_API_KEY=your_gemini_key
```

</td>
<td width="50%">

### 🎛️ **Processing Controls**
- **Strength**: `0.1 - 1.0` modification intensity
- **Creativity**: `0.1 - 1.0` AI imagination level  
- **Formats**: JPEG, PNG, WebP support
- **Seeds**: Reproducible result control

</td>
</tr>
</table>

---

## 📁 **Project Architecture**

```
🏗️ Stable_Diffusion_House_Modifier/
├── 🚀 main.py                      # FastAPI powerhouse
├── 🛠️ image_edit_functions.py      # Core magic functions
├── 📦 requirements.txt             # Dependencies
├── 🎨 templates/                   # Beautiful web interfaces
│   ├── 🏠 index.html              # Main dashboard
│   ├── 🎭 style_transfer.html     # Style magic
│   ├── ✂️ mask.html               # Precision editor
│   └── 📝 text_to_image.html      # Generation hub
├── 🖼️ static/                      # Assets & results
│   └── images/                    # Processed imagery
├── 🏠 House_Images/                # Sample collection
└── 📚 README.md                   # This masterpiece
```

---

## 🌟 **Real-World Applications**

<div align="center">
<table>
<tr>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/80/000000/real-estate.png" alt="Real Estate">
<h3>🏡 Real Estate</h3>
<ul align="left">
<li>Stunning listing visualizations</li>
<li>Seasonal property views</li>
<li>Virtual staging magic</li>
<li>Marketing materials</li>
</ul>
</td>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/80/000000/blueprint.png" alt="Architecture">
<h3>🏗️ Architecture</h3>
<ul align="left">
<li>Conceptual designs</li>
<li>Style exploration</li>
<li>Environment variations</li>
<li>Material previews</li>
</ul>
</td>
<td align="center" width="33%">
<img src="https://img.icons8.com/fluency/80/000000/content.png" alt="Content">
<h3>📱 Content Creation</h3>
<ul align="left">
<li>Social media content</li>
<li>Blog illustrations</li>
<li>Presentation assets</li>
<li>Educational materials</li>
</ul>
</td>
</tr>
</table>
</div>

---

## 🚀 **Deployment Options**

<details>
<summary><strong>🔧 Development Mode</strong></summary>

```bash
# 🛠️ Hot reload for development
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

</details>

<details>
<summary><strong>🏭 Production Ready</strong></summary>

```bash
# ⚡ High-performance production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

</details>

<details>
<summary><strong>🐳 Docker Deployment</strong></summary>

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

</details>

---

## 🔒 **Security & Best Practices**

<div align="center">

| Security Feature | Implementation | Status |
|------------------|----------------|---------|
| 🔐 **API Key Management** | Secure storage, never in code | ✅ Implemented |
| 🛡️ **Input Validation** | Comprehensive sanitization | ✅ Implemented |
| ⚠️ **Error Handling** | User-friendly messages | ✅ Implemented |
| 🚦 **Rate Limiting** | Production consideration | 🔄 Recommended |

</div>

---

## 🤝 **Join Our Community**

<div align="center">

### 🌟 **Contributing**

We welcome contributions with open arms! Here's how to get started:

```bash
# 🍴 Fork the repository
# 🌿 Create your feature branch
git checkout -b feature/amazing-feature

# 💫 Commit your changes  
git commit -m 'Add amazing feature'

# 🚀 Push to the branch
git push origin feature/amazing-feature

# 🎉 Open a Pull Request
```

[![Contributors](https://contrib.rocks/image?repo=yourusername/stable-diffusion-house-modifier)](https://github.com/yourusername/stable-diffusion-house-modifier/graphs/contributors)

</div>

---

## 🏆 **Acknowledgments**

<div align="center">

Special thanks to the incredible teams and technologies that make this possible:

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/60/000000/artificial-intelligence.png" alt="Stability AI">
<br><strong>Stability AI</strong>
<br><em>Stable Diffusion 3.5</em>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/60/000000/google-logo.png" alt="Google AI">
<br><strong>Google AI</strong>
<br><em>Gemini Integration</em>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/60/000000/api-settings.png" alt="FastAPI">
<br><strong>FastAPI</strong>
<br><em>Web Framework</em>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/60/000000/image-editing.png" alt="Pillow">
<br><strong>Pillow</strong>
<br><em>Image Processing</em>
</td>
</tr>
</table>

</div>

---

## 📄 **License**

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

<div align="center">

## 🌈 **Ready to Transform Your House Images?**

<br>

### ⭐ **If this project helped you, please give it a star!** ⭐

<br>

**Made with ❤️ by passionate developers for the AI community**

<br>

<br>

---

**🏠✨ Happy House Image Generation! ✨🏠**

---

</div>
