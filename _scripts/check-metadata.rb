require 'bundler/setup'
require 'jekyll'
require 'nokogiri'
require 'json'
require 'yaml'
require 'date'

source, destination = ARGV.map { |path| File.expand_path(path) }
abort 'Usage: ruby _scripts/check-metadata.rb SOURCE BUILT_SITE' unless source && destination
config = YAML.load_file(File.join(source, '_config.yml'))

def assert(condition, message)
  raise message unless condition
end

def plain_description(value)
  text = value.to_s.gsub(/<!--.*?-->|<[^>]*>/m, '').gsub(/\s+/, ' ').strip
  text.length > 160 ? text[0, 157] + '...' : text
end

posts_by_title = Dir[File.join(source, '_posts', '*.md')].to_h do |path|
  front = File.read(path).split(/^---\s*$\n?/, 3)[1]
  data = YAML.safe_load(front, permitted_classes: [Date, Time])
  [data.fetch('title'), data]
end

post_count = 0
schema_count = 0
Dir[File.join(destination, '**', '*.html')].each do |path|
  doc = Nokogiri::HTML(File.read(path))
  schemas = doc.css('script[type="application/ld+json"]').map { |script| JSON.parse(script.text) }
  schema_count += schemas.length
  schemas.each do |schema|
    assert(schema['@type'] != 'SearchAction', "Unsupported SearchAction in #{path}")
    next unless schema['@type'] == 'WebSite'
    assert(!schema.key?('potentialAction'), "Unsupported search declaration in #{path}")
    assert(schema['name'] == config['title'], "Changed WebSite name in #{path}")
    assert(schema['url'] == config['url'], "Changed WebSite URL in #{path}")
  end
  article = schemas.find { |schema| schema['@type'] == 'BlogPosting' }
  next unless article
  post_count += 1
  front = posts_by_title.fetch(article.fetch('headline'))
  expected = plain_description(front['description'] || front['excerpt'] || config['description'])
  ['meta[name="description"]', 'meta[property="og:description"]', 'meta[name="twitter:description"]'].each do |selector|
    assert(doc.at_css(selector)&.[]('content') == expected, "Wrong #{selector} in #{path}")
  end
  assert(article['description'] == expected, "Wrong JSON-LD description in #{path}")
  assert(article.dig('author', 'name') == '杨钱俊', "Author changed in #{path}")
end
assert(post_count == posts_by_title.size, "Expected #{posts_by_title.size} posts, checked #{post_count}")

manifest = JSON.parse(File.read(File.join(destination, 'manifest.json')))
manifest.fetch('icons').each do |icon|
  assert(File.file?(File.join(destination, icon.fetch('src').delete_prefix('/'))), "Missing manifest icon #{icon['src']}")
end

# Render the actual template heads with edge cases without writing fixture posts.
site = Jekyll::Site.new(Jekyll.configuration('source' => source, 'destination' => destination, 'quiet' => true))
head_source = File.read(File.join(source, '_layouts', 'default.html')).split('</head>').first
post_source = File.read(File.join(source, '_layouts', 'post.html')).split('<!-- 文章头部 -->').first.sub(/\A---.*?---\s*/m, '')
fixtures = [
  {'title' => 'Quotes " & backslash \\ </script><script>should-not-run</script>', 'excerpt' => "Quoted \"text\" & paths \\alpha\nnext line", 'description' => nil},
  {'title' => 'Explicit description', 'excerpt' => 'Not the selected summary', 'description' => 'Preferred "description" & detail'},
  {'title' => 'HTML excerpt', 'excerpt' => '<p>A <strong>formatted</strong> excerpt</p>', 'description' => nil},
  {'title' => 'Long excerpt', 'excerpt' => '字' * 200, 'description' => nil},
  {'title' => 'Site fallback', 'excerpt' => nil, 'description' => nil}
]
fixtures.each do |fixture|
  page = fixture.merge('layout' => 'post', 'date' => Time.new(2026, 10, 30), 'url' => '/fixture/')
  context = {'site' => config, 'page' => page}
  html = Liquid::Template.parse(head_source + '</head><body>' + post_source + '</body></html>').render!(context, registers: {site: site})
  doc = Nokogiri::HTML(html)
  schemas = doc.css('script[type="application/ld+json"]').map { |script| JSON.parse(script.text) }
  assert(schemas.size == 3, "Schema/script boundary broken for #{fixture['title']}")
  article = schemas.find { |schema| schema['@type'] == 'BlogPosting' }
  expected = plain_description(fixture['description'] || fixture['excerpt'] || config['description'])
  assert(article['headline'] == fixture['title'], 'JSON-LD headline changed by escaping')
  assert(article['description'] == expected, 'JSON-LD description fixture mismatch')
  assert(doc.at_css('meta[name="description"]')['content'] == expected, 'Meta fixture mismatch')
  assert(doc.at_css('meta[property="og:title"]')['content'] == fixture['title'], 'OG title fixture mismatch')
  assert(doc.css('script').none? { |script| script.text == 'should-not-run' }, 'Closing script injection was not escaped')
end

puts JSON.pretty_generate({status: 'PASS', posts_checked: post_count, json_ld_blocks_checked: schema_count, description_fields_per_post: 4, edge_case_fixtures: fixtures.size, manifest_icons_checked: manifest['icons'].size, unsupported_search_action: false})
