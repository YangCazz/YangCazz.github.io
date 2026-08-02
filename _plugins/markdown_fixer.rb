# 构建期内容修复 —— 把原本运行时的 LaTeX→Unicode 映射、classDef 注入、
# 表格 <em> 修复下沉到构建期,减少每个访客浏览器的 JS 解析开销。
#
# 两个 hook:
#   1. :pre_render —— 修改 markdown 源:fenced mermaid 块做 LaTeX→Unicode 清理
#      + 注入六种 classDef(与 CLAUDE.md 色板一致)
#   2. :post_render —— 修改 HTML 输出:修复 kramdown 表格内 $...<em>...</em>$
#
# 边界:节点对比度修正(fixTextContrast)依赖 Mermaid 运行时 SVG,保留在前端。

Jekyll::Hooks.register :posts, :pre_render do |post|
  content = post.content.to_s
  next if content.empty?

  # 匹配 fenced mermaid 块
  content.gsub!(/```mermaid\s*\n(.*?)```/m) do |_m|
    src = Regexp.last_match(1)
    src = sanitize_math_for_mermaid(src)
    src = inject_class_defs(src)
    "```mermaid\n#{src}```"
  end

  post.content = content
end

Jekyll::Hooks.register :posts, :post_render do |post, output|
  src = (output || post.output).to_s
  # 只修复 kramdown 表格单元格(<td>/<th>)内 $..._...$ 的 _ 被 kramdown 转成
  # <em> 后破坏 MathJax 的问题 —— 作用域严格限定在 td/th 内(与原运行时
  # document.querySelectorAll('td, th') 行为一致)。
  # 警告:此前的全局 gsub 版本会误伤正文段落中的行内公式(正文里 kramdown 的 _
  # 强调也可能产生 <em>,导致 $...$ 被错误吞并为 \(...\)),已回归为限定 td/th。
  src = src.gsub(/(<t[dh][^>]*>)(.*?)(<\/t[dh]>)/m) do |_m|
    cell_open = Regexp.last_match(1)
    inner = Regexp.last_match(2)
    cell_close = Regexp.last_match(3)
    prev = nil
    while prev != inner
      prev = inner
      inner = inner.gsub(
        /\$([^$]*?)<em>([^<$]*?)<\/em>([^$]*?)\$/,
        '\\(\1\2\3\\)'
      )
    end
    cell_open + inner + cell_close
  end
  post.output = src
end

CLASS_DEFS = [
  'classDef core fill:#1a237e,stroke:#4299e1,color:#e8edf5,stroke-width:2.5px',
  'classDef mid fill:#1b2d3a,stroke:#667eea,color:#e8edf5,stroke-width:2px',
  'classDef proc fill:#1a1f3a,stroke:#7c3aed,color:#e8edf5,stroke-width:2px',
  'classDef out fill:#1a2a1a,stroke:#48bb78,color:#e8edf5,stroke-width:2.5px',
  'classDef hl fill:#2a1a2e,stroke:#ed64a6,color:#e8edf5,stroke-width:2.5px',
  'classDef dim fill:#2d3748,stroke:#718096,color:#cbd5e0,stroke-width:1.5px'
].join("\n")

def inject_class_defs(text)
  if text.match?(/^(\s*(?:graph|flowchart)\s+(?:LR|RL|TB|TD|BT).*)$/) && !text.include?('classDef')
    text.sub(/^(\s*(?:graph|flowchart)\s+(?:LR|RL|TB|TD|BT).*)$/) do |line|
      line + "\n" + CLASS_DEFS
    end
  else
    text
  end
end

def sanitize_math_for_mermaid(text)
  greek = {
    '\\alpha' => 'α', '\\beta' => 'β', '\\gamma' => 'γ', '\\delta' => 'δ',
    '\\epsilon' => 'ε', '\\theta' => 'θ', '\\lambda' => 'λ', '\\mu' => 'µ',
    '\\pi' => 'π', '\\rho' => 'ρ', '\\sigma' => 'σ', '\\tau' => 'τ',
    '\\phi' => 'φ', '\\omega' => 'ω', '\\Delta' => 'Δ', '\\Gamma' => 'Γ',
    '\\Omega' => 'Ω', '\\Sigma' => 'Σ', '\\Xi' => 'Ξ', '\\Pi' => 'Π',
    '\\times' => '×', '\\cdot' => '·', '\\div' => '÷', '\\pm' => '±',
    '\\leq' => '≤', '\\geq' => '≥', '\\neq' => '≠', '\\approx' => '≈',
    '\\infty' => '∞', '\\partial' => '∂', '\\nabla' => '∇',
    '\\rightarrow' => '→', '\\Rightarrow' => '⇒', '\\leftarrow' => '←',
    '\\cap' => '∩', '\\cup' => '∪', '\\subset' => '⊂', '\\supseteq' => '⊇',
    '\\in' => '∈', '\\notin' => '∉', '\\wedge' => '∧', '\\vee' => '∨',
    '\\forall' => '∀', '\\exists' => '∃', '\\neg' => '¬',
    '\\mathbb{R}' => 'ℝ', '\\mathbb{N}' => 'ℕ', '\\mathbb{Z}' => 'ℤ',
    '\\mathcal{L}' => 'ℒ', '\\mathcal{l}' => 'ℓ',
    '\\ldots' => '…', '\\cdots' => '⋯', '\\vdots' => '⋮',
    '\\quad' => ' ', '\\,' => '', '\\;' => '', '\\:' => '', '\\!' => ''
  }
  num_sub = { '0' => '₀', '1' => '₁', '2' => '₂', '3' => '₃', '4' => '₄',
              '5' => '₅', '6' => '₆', '7' => '₇', '8' => '₈', '9' => '₉' }
  num_sup = { '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴',
              '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹' }
  latin_sub = { 'a' => 'ₐ', 'e' => 'ₑ', 'i' => 'ᵢ', 'n' => 'ₙ', 'o' => 'ₒ',
                'r' => 'ᵣ', 's' => 'ₛ', 't' => 'ₜ', 'u' => 'ᵤ', 'v' => 'ᵥ',
                'x' => 'ₓ' }

  result = text.gsub(/\$([^$]+)\$/) do |_m|
    math = Regexp.last_match(1)
    greek.each { |k, v| math = math.gsub(k, v) }
    math = math.gsub(/_\{([^}]+)\}/) do |_mm|
      chars = Regexp.last_match(1)
      if chars.chars.all? { |c| num_sub[c] || latin_sub[c] }
        chars.chars.map { |c| num_sub[c] || latin_sub[c] }.join
      else
        '_' + chars
      end
    end
    math = math.gsub(/\^\{([^}]+)\}/) do |_mm|
      chars = Regexp.last_match(1)
      if chars.chars.all? { |c| num_sup[c] }
        chars.chars.map { |c| num_sup[c] }.join
      else
        '^' + chars
      end
    end
    math = math.gsub(/\\text\{([^}]*)\}/, '\1')
    math = math.gsub(/\\mathrm\{([^}]*)\}/, '\1')
    math = math.gsub(/\\(log|ln|sin|cos|exp|max|min|det|lim|arg)\b/, '\1')
    math = math.gsub(/\$/, '')
    math = math.gsub(/[{}]/, '')
    math.strip
  end
  result
end
