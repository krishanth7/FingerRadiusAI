# frozen_string_literal: true

# conformance_test.rb - Check the Ruby port matches Python exactly.
#
# Run:  ruby ports/ruby/conformance_test.rb

require 'json'
require_relative 'finger_radius'

FIXTURE = File.expand_path('../fixtures/conformance.json', __dir__)
TOLERANCE = 1e-6

fixture = JSON.parse(File.read(FIXTURE))
failures = []
checks = 0

recognizer = FingerRadius::GestureRecognizer.new

fixture['gestures'].each do |item|
  landmarks = item['landmarks']
  expected = item['expected']
  actual = recognizer.classify(landmarks)
  checks += 1

  if actual[:gesture] != expected['gesture']
    failures << "#{item['id']}: gesture #{actual[:gesture]} != #{expected['gesture']}"
  end
  if actual[:extended_count] != expected['extended_count']
    failures << "#{item['id']}: extended_count #{actual[:extended_count]} != #{expected['extended_count']}"
  end
  expected['fingers'].each do |finger, want|
    got = actual[:fingers][finger]
    failures << "#{item['id']}: finger #{finger} #{got} != #{want}" if got != want
  end

  radii = FingerRadius.radii(landmarks)
  expected['radii'].each do |pair, want|
    got = radii[pair]
    checks += 1
    failures << "#{item['id']}: radius #{pair} #{got} != #{want}" if (got - want).abs > TOLERANCE
  end
end

kalman = fixture['kalman']
filter = FingerRadius::KalmanFilter1D.new(
  process_noise: kalman['process_noise'],
  measurement_noise: kalman['measurement_noise']
)
kalman['input'].each_with_index do |value, i|
  got = filter.update(value)
  want = kalman['expected'][i]
  checks += 1
  failures << "kalman[#{i}]: #{got} != #{want} (delta #{(got - want).abs})" if (got - want).abs > TOLERANCE
end

if failures.empty?
  puts "ruby: PASS - #{checks} checks against the Python fixture"
  exit 0
else
  puts "ruby: FAIL - #{failures.length} of #{checks} checks"
  failures.first(10).each { |f| puts "  #{f}" }
  exit 1
end
